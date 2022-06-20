import numpy as np
import torch
import cv2
import time
import copy

from trackron.structures.tracklet import Tracklet, BaseTrack, TrackState
from trackron.data.utils import resize_image,  sample_target_brpadding, normalize_boxes

from .utils.matching import ious, iou_distance, linear_assignment, fuse_score
from .build import TRACKER_REGISTRY
from .siamese_tracker import SiameseTracker
from .base_tracker import PublicPostProcess, PostProcess
from .utils.kalman_filter import KalmanFilter
from .utils.preprocessing import numpy_to_torch


class STrack(BaseTrack):
  shared_kalman = KalmanFilter()

  def __init__(self, tlwh, score):

    # wait activate
    self._tlwh = np.asarray(tlwh, dtype=np.float)
    self.kalman_filter = None
    self.mean, self.covariance = None, None
    self.is_activated = False

    self.score = score
    self.tracklet_len = 0

  def predict(self):
    mean_state = self.mean.copy()
    if self.state != TrackState.Tracked:
      mean_state[7] = 0
    self.mean, self.covariance = self.kalman_filter.predict(
        mean_state, self.covariance)

  @staticmethod
  def multi_predict(stracks):
    if len(stracks) > 0:
      multi_mean = np.asarray([st.mean.copy() for st in stracks])
      multi_covariance = np.asarray([st.covariance for st in stracks])
      for i, st in enumerate(stracks):
        if st.state != TrackState.Tracked:
          multi_mean[i][7] = 0
      multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
          multi_mean, multi_covariance)
      for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        stracks[i].mean = mean
        stracks[i].covariance = cov

  def activate(self, kalman_filter, frame_num):
    """Start a new tracklet"""
    self.kalman_filter = kalman_filter
    self.track_id = self.next_id()
    self.mean, self.covariance = self.kalman_filter.initiate(
        self.tlwh_to_xyah(self._tlwh))

    self.tracklet_len = 0
    self.state = TrackState.Tracked
    if frame_num == 1:
      self.is_activated = True
    # self.is_activated = True
    self.frame_num = frame_num
    self.start_frame = frame_num

  def re_activate(self, new_track, frame_num, new_id=False):
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
    self.tracklet_len = 0
    self.state = TrackState.Tracked
    self.is_activated = True
    self.frame_num = frame_num
    if new_id:
      self.track_id = self.next_id()
    self.score = new_track.score

  def update(self, new_track, frame_num):
    """
        Update a matched track
        :type new_track: STrack
        :type frame_num: int
        :type update_feature: bool
        :return:
        """
    self.frame_num = frame_num
    self.tracklet_len += 1

    new_tlwh = new_track.tlwh
    self.mean, self.covariance = self.kalman_filter.update(
        self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
    self.state = TrackState.Tracked
    self.is_activated = True

    self.score = new_track.score

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

  def __repr__(self):
    return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame,
                                  self.end_frame)


@TRACKER_REGISTRY.register()
class ByteTracker(SiameseTracker):

  def setup(self):
    ### MOT parameters
    super().setup()
    self.num_classes = self.cfg.MODEL.NUM_CLASS
    self.max_time_lost = 30
    self.max_age = 30
    self.score_thresh = self.det_thresh + 0.1
    self.resize = (800, 1440)
    self.post_process = PostProcess(image_sz=self.resize, box_fmt='xyxy', box_absolute=True, uni_scale=True)

  def reset_mot(self, image):
    self.tracked_stracks = []
    self.lost_stracks = []
    self.removed_stracks = []
    self.kalman_filter = KalmanFilter()

    self.tracks_dict = dict()
    self.tracks = list()
    self.unmatched_tracks = list()
    self.ori_image_szs = torch.tensor([image.shape[:2]],
                                      dtype=torch.float32,
                                      device=self.device)

    # self.resize = get_size_with_aspect_ratio(image.shape[:2])[::-1]
    # self.resize = (image.shape[1], image.shape[0])

  def init_mot(self, image, info: dict, category=None) -> dict:
    # Time initialization
    tic = time.time()
    self.reset_mot(image)
    self.category = category

    return self.track_mot(image, info)

  def track_mot(self, image, info):
    rz_image = resize_image(image, self.resize)
    im_tensor = numpy_to_torch(rz_image).to(self.device) / 255.0
    mask_tensor = torch.zeros((1, *im_tensor.shape[-2:]),
                              dtype=torch.bool,
                              device=self.device)
    if self.use_pub_detection:
      ### not supported currently
      public_detections = torch.tensor(info['init_detections']).to(self.device)
      public_detections[:, :4] = normalize_boxes(public_detections[:, :4],
                                                 image.shape[:2],
                                                 in_format='xywh',
                                                 out_format='cxcywh')
      self.ref_info['public_detections'] = public_detections.unsqueeze(
          0)  ### batch dim
      det_boxes = self.ref_info['public_detections'][..., :4]
      det_logits = self.ref_info['public_detections'][..., 4]
    with torch.no_grad():
      outputs = self.net.track_mot(im_tensor)
      
    results = self.post_process(outputs, self.ori_image_szs, self.category)
    res_track = self.update_mot(results[0])

    return res_track, results[0]

  def update_mot(self, output_results, use_embedding=False):
    scores = output_results["scores"].cpu().numpy()
    bboxes = output_results["boxes"].cpu().numpy()  # x1y1x2y2
 
    remain_inds = scores > self.det_thresh
    inds_low = scores > 0.1
    inds_high = scores < self.det_thresh

    inds_second = np.logical_and(inds_low, inds_high)
    dets_second = bboxes[inds_second]
    dets = bboxes[remain_inds]
    scores_keep = scores[remain_inds]
    scores_second = scores[inds_second]

    activated_starcks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []

    if len(dets) > 0:
      detections = [
          STrack(STrack.tlbr_to_tlwh(box), score)
          for box, score in zip(dets, scores_keep)
      ]
    else:
      detections = []
    ''' Add newly detected stracks to tracked_stracks'''
    unconfirmed = []
    tracked_stracks = []
    for track in self.tracked_stracks:
      if not track.is_activated:
        unconfirmed.append(track)
      else:
        tracked_stracks.append(track)
    ''' Step 2: First association, with high score detection boxes'''
    strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    # Predict the current location with KF
    STrack.multi_predict(strack_pool)
    dists = iou_distance(strack_pool, detections)
    dists = fuse_score(dists, detections)
    matches, u_track, u_detection = linear_assignment(dists, thresh=0.9)

    for itracked, idet in matches:
      track = strack_pool[itracked]
      det = detections[idet]
      if track.state == TrackState.Tracked:
        track.update(detections[idet], self.frame_num)
        activated_starcks.append(track)
      else:
        track.re_activate(det, self.frame_num, new_id=False)
        refind_stracks.append(track)

    ''' Step 3: Second association, with low score detection boxes'''
    # association the untrack to the low score detections
    if len(dets_second) > 0:
      '''Detections'''
      detections_second = [
          STrack(STrack.tlbr_to_tlwh(tlbr), s)
          for (tlbr, s) in zip(dets_second, scores_second)
      ]
    else:
      detections_second = []
    r_tracked_stracks = [
        strack_pool[i]
        for i in u_track
        if strack_pool[i].state == TrackState.Tracked
    ]
    dists = iou_distance(r_tracked_stracks, detections_second)
    matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
    for itracked, idet in matches:
      track = r_tracked_stracks[itracked]
      det = detections_second[idet]
      if track.state == TrackState.Tracked:
        track.update(det, self.frame_num)
        activated_starcks.append(track)
      else:
        track.re_activate(det, self.frame_num, new_id=False)
        refind_stracks.append(track)

    for it in u_track:
      track = r_tracked_stracks[it]
      if not track.state == TrackState.Lost:
        track.mark_lost()
        lost_stracks.append(track)
    '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
    detections = [detections[i] for i in u_detection]
    dists = iou_distance(unconfirmed, detections)
    dists = fuse_score(dists, detections)
    matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
    for itracked, idet in matches:
      unconfirmed[itracked].update(detections[idet], self.frame_num)
      activated_starcks.append(unconfirmed[itracked])
    for it in u_unconfirmed:
      track = unconfirmed[it]
      track.mark_removed()
      removed_stracks.append(track)
    """ Step 4: Init new stracks"""
    for inew in u_detection:
      track = detections[inew]
      if track.score < self.score_thresh:
        continue
      track.activate(self.kalman_filter, self.frame_num)
      activated_starcks.append(track)
    """ Step 5: Update state"""
    for track in self.lost_stracks:
      if self.frame_num - track.end_frame > self.max_age:
        track.mark_removed()
        removed_stracks.append(track)

    # print('Ramained match {} s'.format(t4-t3))

    self.tracked_stracks = [
        t for t in self.tracked_stracks if t.state == TrackState.Tracked
    ]
    self.tracked_stracks = joint_stracks(self.tracked_stracks,
                                         activated_starcks)
    self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
    self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
    self.removed_stracks.extend(removed_stracks)
    self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
        self.tracked_stracks, self.lost_stracks)
    # get scores of lost tracks
    # output_stracks = [
    #     track for track in self.tracked_stracks if track.is_activated
    # ]
    output_stracks = []
    for track in self.tracked_stracks:
      if track.is_activated:
        track_obj = {
            'tracking_id': track.track_id,
            'bbox': track.tlbr,
            'score': track.score,
            'active': 1.0,
        }
        output_stracks.append(track_obj)

    return output_stracks


def joint_stracks(tlista, tlistb):
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


def sub_stracks(tlista, tlistb):
  stracks = {}
  for t in tlista:
    stracks[t.track_id] = t
  for t in tlistb:
    tid = t.track_id
    if stracks.get(tid, 0):
      del stracks[tid]
  return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
  pdist = iou_distance(stracksa, stracksb)
  pairs = np.where(pdist < 0.15)
  dupa, dupb = list(), list()
  for p, q in zip(*pairs):
    timep = stracksa[p].frame_num - stracksa[p].start_frame
    timeq = stracksb[q].frame_num - stracksb[q].start_frame
    if timep > timeq:
      dupb.append(q)
    else:
      dupa.append(p)
  resa = [t for i, t in enumerate(stracksa) if not i in dupa]
  resb = [t for i, t in enumerate(stracksb) if not i in dupb]
  return resa, resb
