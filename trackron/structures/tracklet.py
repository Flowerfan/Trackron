import torch
import numpy as np
from collections import OrderedDict, deque
from trackron.trackers.utils.kalman_filter import KalmanFilter
import trackron.trackers.utils.matching as matching
import torch.nn.functional as F


def tlbr_to_xyah(tlbr):
  """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
  ret = np.asarray(tlbr).copy()
  w = tlbr[2] - tlbr[0]
  h = tlbr[3] - tlbr[1]
  ret[:2] = (ret[:2] + ret[2:]) / 2
  ret[2] = w / (h + 1e-6)
  ret[3] = h
  return ret


class TrackState(object):
  New = 0
  Tracked = 1
  Lost = 2
  Removed = 3


class BaseTrack(object):
  _count = 0

  track_id = 0
  is_activated = False
  state = TrackState.New

  history = OrderedDict()
  features = []
  curr_feature = None
  score = 0
  start_frame = 0
  frame_id = 0
  time_since_update = 0

  # multi-camera
  location = (np.inf, np.inf)

  @property
  def end_frame(self):
    return self.frame_id

  @staticmethod
  def next_id():
    BaseTrack._count += 1
    return BaseTrack._count

  def activate(self, *args):
    raise NotImplementedError

  def predict(self):
    raise NotImplementedError

  def update(self, *args, **kwargs):
    raise NotImplementedError

  def mark_lost(self):
    self.state = TrackState.Lost

  def mark_removed(self):
    self.state = TrackState.Removed

  @property
  # @jit(nopython=True)
  def tlwh(self):
    """Get current position in bounding box format `(top left x, top left y,
              width, height)`.
      """
    if self.mean is None:
      return self._tlwh.copy()
    ret = self.mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

  @property
  # @jit(nopython=True)
  def tlbr(self):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
      `(top left, bottom right)`.
      """
    ret = self.tlwh.copy()
    ret[2:] += ret[:2]
    return ret

  @staticmethod
  # @jit(nopython=True)
  def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
      height)`, where the aspect ratio is `width / height`.
      """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

  def to_xyah(self):
    return self.tlwh_to_xyah(self.tlwh)

  @staticmethod
  # @jit(nopython=True)
  def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret

  @staticmethod
  # @jit(nopython=True)
  def tlwh_to_tlbr(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret


class Tracklet(BaseTrack):
  shared_kalman = KalmanFilter()

  def __init__(self,
               tlbr,
               score,
               temp_feat,
               buffer_size=30,
               mask=None,
               pose=None,
               ac=False,
               category=-1,
               use_kalman=True):

    # wait activate
    self._tlbr = np.asarray(tlbr, dtype=np.float)
    self.kalman_filter = None
    self.mean, self.covariance = None, None
    self.use_kalman = use_kalman
    if not use_kalman:
      ac = True
    self.is_activated = ac

    self.score = score
    self.category = category
    self.tracklet_len = 0

    self.smooth_feat = None
    self.update_features(temp_feat)
    self.features = deque([], maxlen=buffer_size)
    self.alpha = 0.9
    self.mask = mask
    self.pose = pose

  def update_features(self, feat):
    self.curr_feat = feat
    if self.smooth_feat is None:
      self.smooth_feat = feat
    elif self.smooth_feat.shape == feat.shape:
      self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
    else:
      pass

  def predict(self):
    mean_state = self.mean.copy()
    if self.state != TrackState.Tracked:
      mean_state[7] = 0
    self.mean, self.covariance = self.kalman_filter.predict(
        mean_state, self.covariance)

  @staticmethod
  def multi_predict(tracklets):
    if len(tracklets) > 0:
      multi_mean = np.asarray([st.mean.copy() for st in tracklets])
      multi_covariance = np.asarray([st.covariance for st in tracklets])
      for i, st in enumerate(tracklets):
        if st.state != TrackState.Tracked:
          multi_mean[i][7] = 0
      multi_mean, multi_covariance = Tracklet.shared_kalman.multi_predict(
          multi_mean, multi_covariance)
      for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        tracklets[i].mean = mean
        tracklets[i].covariance = cov

  def activate(self, kalman_filter, frame_id):
    """Start a new tracklet"""
    self.kalman_filter = kalman_filter
    self.track_id = self.next_id()
    self.mean, self.covariance = self.kalman_filter.initiate(
        tlbr_to_xyah(self._tlbr))

    self.tracklet_len = 0
    self.state = TrackState.Tracked
    if frame_id == 1:
      self.is_activated = True
    #self.is_activated = True
    self.frame_id = frame_id
    self.start_frame = frame_id

  def re_activate(self, new_track, frame_id, new_id=False, update_feature=True):
    if self.use_kalman:
      self.mean, self.covariance = self.kalman_filter.update(
          self.mean, self.covariance, tlbr_to_xyah(new_track.tlbr))
    else:
      self.mean, self.covariance = None, None
      self._tlbr = np.asarray(new_track.tlbr, dtype=np.float)
    if update_feature:
      self.update_features(new_track.curr_feat)
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_id = frame_id
    if new_id:
      self.track_id = self.next_id()
    if not new_track.mask is None:
      self.mask = new_track.mask

  def update(self, new_track, frame_id, update_feature=True):
    """
        Update a matched track
        :type new_track: Tracklet
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
    self.frame_id = frame_id
    self.tracklet_len += 1

    new_tlbr = new_track.tlbr
    if self.use_kalman:
      self.mean, self.covariance = self.kalman_filter.update(
          self.mean, self.covariance, tlbr_to_xyah(new_tlbr))
    else:
      self.mean, self.covariance = None, None
      self._tlbr = np.asarray(new_tlbr, dtype=np.float)
    self.state = TrackState.Tracked
    self.is_activated = True

    self.score = new_track.score
    '''
        For TAO dataset 
        '''
    self.category = new_track.category
    if update_feature:
      self.update_features(new_track.curr_feat)
    if not new_track.mask is None:
      self.mask = new_track.mask
    if not new_track.pose is None:
      self.pose = new_track.pose

  @property
  def tlbr(self):
    """Get current position in bounding box format `(l,t,r,b)`.
        """
    if self.mean is None:
      return self._tlbr.copy()
    ret = self.mean[:4].copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    ret[2:] += ret[:2]
    return ret

  def to_xyah(self):
    return tlbr_to_xyah(self.tlbr)

  def __repr__(self):
    return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame,
                                  self.end_frame)


def joint_tracklets(tlista, tlistb):
  exists = {}
  res = []
  for t in tlista:
    exists[t.track_id] = 1
    res.append(t)
  for t in tlistb:
    tid = t.track_id
    if not exists.get(tid, 0):
      exists[tid] = 1
      res.append(t)
  return res


def sub_tracklets(tlista, tlistb):
  tracklets = {}
  for t in tlista:
    tracklets[t.track_id] = t
  for t in tlistb:
    tid = t.track_id
    if tracklets.get(tid, 0):
      del tracklets[tid]
  return list(tracklets.values())


def remove_duplicate_tracklets(trackletsa, trackletsb, ioudist=0.15):
  pdist = matching.iou_distance(trackletsa, trackletsb)
  pairs = np.where(pdist < ioudist)
  dupa, dupb = list(), list()
  for p, q in zip(*pairs):
    timep = trackletsa[p].frame_id - trackletsa[p].start_frame
    timeq = trackletsb[q].frame_id - trackletsb[q].start_frame
    if timep > timeq:
      dupb.append(q)
    else:
      dupa.append(p)
  resa = [t for i, t in enumerate(trackletsa) if not i in dupa]
  resb = [t for i, t in enumerate(trackletsb) if not i in dupb]
  return resa, resb
