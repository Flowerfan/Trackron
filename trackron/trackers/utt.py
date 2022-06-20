import numpy as np
import torch
import cv2
import time

from trackron.structures.tracklet import BaseTrack, TrackState, joint_tracklets, remove_duplicate_tracklets, sub_tracklets
from trackron.data.utils import get_size_with_aspect_ratio, normalize_boxes

from .utils.matching import fuse_score, iou_distance, linear_assignment
from .build import TRACKER_REGISTRY
from .siamese_tracker import SiameseTracker, clip_box
from .base_tracker import PublicPostProcess, PostProcess
from .utils.kalman_filter import KalmanFilter
from .utils.preprocessing import numpy_to_torch


class STrack(BaseTrack):
  shared_kalman = KalmanFilter()

  def __init__(self, tlwh, score, temp_feat):

    # wait activate
    self._tlwh = np.asarray(tlwh, dtype=np.float)
    self.kalman_filter = None
    self.mean, self.covariance = None, None
    self.track_box = None
    self.is_activated = False

    self.score = score
    self.tracklet_len = 0

    self.smooth_feat = None
    self.update_features(temp_feat)
    # self.features = deque([], maxlen=buffer_size)
    self.alpha = 0.9

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

  def update(self, new_track, frame_num, update_feature=True):
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

    if update_feature:
      self.update_features(new_track.curr_feat)

  def update_trackbox(self, track_box):
    self.track_box = track_box

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

  @property
  # @jit(nopython=True)
  def track_tlbr(self):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
    if self.track_box is None:
      return self.tlbr
    else:
      return self.track_box.copy()

  @property
  # @jit(nopython=True)
  def cxcywh(self):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
    ret = self.tlwh.copy()
    ret[:2] += ret[2:] * 0.5
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
class UTTracker(SiameseTracker):

  def reset_mot(self):
    self.tracked_stracks = []
    self.lost_stracks = []
    self.removed_stracks = []
    self.last_detect_tracks = []
    self.kalman_filter = KalmanFilter()
    self.mot_ref_info = {}

  def init_mot(self, image, info: dict, category=None) -> dict:
    self.reset_mot()
    self.score_thresh = self.det_thresh
    self.max_age = 30
    self.category = category
    if self.use_pub_detection:
      self.post_process = PublicPostProcess()
    else:
      self.post_process = PostProcess()

    # Time initialization
    tic = time.time()
    self.resize = get_size_with_aspect_ratio(image.shape[:2])
    # self.resize = (608, 1088)
    # self.resize = (image.shape[0], image.shape[1])
    self.ori_image_szs = torch.tensor([image.shape[:2]],
                                      dtype=torch.float32,
                                      device=self.device)
    return self.track_mot(image, info)

  def track_mot(self, image, info):

    resize_image = cv2.resize(image, self.resize[::-1])
    im_tensor = numpy_to_torch(resize_image).to(self.device) / 255.0
    mask_tensor = torch.zeros((1, *im_tensor.shape[-2:]),
                              dtype=torch.bool,
                              device=self.device)
    if self.use_pub_detection:
      public_detections = torch.tensor(info['init_detections']).to(self.device)
      public_detections[:, :4] = normalize_boxes(public_detections[:, :4],
                                                 image.shape[:2],
                                                 in_format='xywh',
                                                 out_format='cxcywh')
      self.mot_ref_info['public_detections'] = public_detections.unsqueeze(
          0)  ### batch dimc
    # Predict the current location with KF
    if self.frame_num > 1 and len(self.last_detect_tracks) > 0:
      STrack.multi_predict(self.last_detect_tracks)
      for track in self.last_detect_tracks:
        track.update_trackbox(None)
      proposal = torch.tensor([t.tlwh for t in self.last_detect_tracks],
                              dtype=torch.float32).to(self.device)
      target_feature = torch.tensor(
          [t.curr_feat for t in self.last_detect_tracks],
          dtype=torch.float32).to(self.device)
      self.mot_ref_info['proposal'] = normalize_boxes(proposal,
                                                      image.shape[:2],
                                                      in_format='xywh',
                                                      out_format='cxcywh')[None]
      self.mot_ref_info['target_feat'] = target_feature[None]
      assert len(self.last_detect_tracks) == len(target_feature)

    outputs, self.mot_ref_info = self.net.track_mot(im_tensor,
                                                    mask=mask_tensor,
                                                    ref_info=self.mot_ref_info)
    results = self.post_process(outputs, self.ori_image_szs, self.category)
    res_track = self.update_mot(results[0])
    return res_track, results[0]

  def update_mot(self, output_results, use_embedding=False):
    scores = output_results["scores"].cpu().numpy()
    bboxes = output_results["boxes"].cpu().numpy()  # x1y1x2y2
    features = output_results["embs"].cpu().numpy()
    track_bboxes = output_results["track_boxes"].cpu().numpy(
    ) if "track_boxes" in output_results else None  # x1y1x2y2
    track_scores = output_results["track_scores"].cpu().numpy(
    ) if "track_scores" in output_results else None  # x1y1x2y2
    for idx, track in enumerate(self.last_detect_tracks):
      if track_scores[idx] > 0.9:
        track.update_trackbox(track_bboxes[idx])

    remain_inds = scores > self.score_thresh
    inds_low = scores > 0.2
    inds_high = scores < self.score_thresh

    inds_second = np.logical_and(inds_low, inds_high)
    dets_second = bboxes[inds_second]
    dets = bboxes[remain_inds]
    feats = features[remain_inds]
    feats_second = features[inds_second]
    scores_keep = scores[remain_inds]
    scores_second = scores[inds_second]

    activated_starcks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []
    detect_tracks = []

    if len(dets) > 0:
      detections = [
          STrack(STrack.tlbr_to_tlwh(box), score, feat)
          for box, score, feat in zip(dets, scores_keep, feats)
      ]
    else:
      detections = []
    ''' Add newly detected stracks to tracked_stracks'''
    unconfirmed = []
    tracked_stracks = []
    for idx, track in enumerate(self.tracked_stracks):
      if not track.is_activated:
        unconfirmed.append(track)
      else:
        tracked_stracks.append(track)
    ''' Step 2: First association, with high score detection boxes'''
    STrack.multi_predict(self.lost_stracks)
    strack_pool = joint_tracklets(tracked_stracks, self.lost_stracks)
    # STrack.multi_predict(strack_pool)
    # Predict the current location with KF
    track_boxes = np.array([t.track_tlbr for t in strack_pool])
    det_boxes = np.array([d.tlbr for d in detections])
    dists = iou_distance(track_boxes, det_boxes)
    # dists = iou_distance(strack_pool, detections)
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
      detect_tracks.append(track)
    ''' Step 3: Second association, with low score detection boxes'''
    # association the untrack to the low score detections
    if len(dets_second) > 0:
      '''Detections'''
      detections_second = [
          STrack(STrack.tlbr_to_tlwh(tlbr), s, feat)
          for (tlbr, s, feat) in zip(dets_second, scores_second, feats_second)
      ]
    else:
      detections_second = []
    r_tracked_stracks = [
        strack_pool[i]
        for i in u_track
        if strack_pool[i].state == TrackState.Tracked
    ]
    track_boxes = np.array([t.track_tlbr for t in r_tracked_stracks])
    det_boxes = np.array([d.tlbr for d in detections_second])
    dists = iou_distance(track_boxes, det_boxes)
    # dists = iou_distance(r_tracked_stracks, detections_second)
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
      detect_tracks.append(track)

    for it in u_track:
      track = r_tracked_stracks[it]
      if not track.state == TrackState.Lost:
        track.mark_lost()
        lost_stracks.append(track)
    '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
    rest_detections = [detections[i] for i in u_detection]
    dists = iou_distance(unconfirmed, rest_detections)
    dists = fuse_score(dists, rest_detections)
    matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
    for itracked, idet in matches:
      unconfirmed[itracked].update(rest_detections[idet], self.frame_num)
      activated_starcks.append(unconfirmed[itracked])
      detect_tracks.append(unconfirmed[itracked])
    for it in u_unconfirmed:
      track = unconfirmed[it]
      track.mark_removed()
      removed_stracks.append(track)
    """ Step 4: Init new stracks"""
    for inew in u_detection:
      track = rest_detections[inew]
      if track.score < self.det_thresh + 0.1:
        continue
      track.activate(self.kalman_filter, self.frame_num)
      activated_starcks.append(track)
      detect_tracks.append(track)
    """ Step 5: Update state"""
    for track in self.lost_stracks:
      if self.frame_num - track.end_frame > self.max_age:
        track.mark_removed()
        removed_stracks.append(track)

    # print('Ramained match {} s'.format(t4-t3))

    self.tracked_stracks = [
        t for t in self.tracked_stracks if t.state == TrackState.Tracked
    ]
    self.tracked_stracks = joint_tracklets(self.tracked_stracks,
                                         activated_starcks)
    self.tracked_stracks = joint_tracklets(self.tracked_stracks, refind_stracks)
    self.lost_stracks = sub_tracklets(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = sub_tracklets(self.lost_stracks, self.removed_stracks)
    self.removed_stracks.extend(removed_stracks)
    self.tracked_stracks, self.lost_stracks = remove_duplicate_tracklets(
        self.tracked_stracks, self.lost_stracks)
    self.last_detect_tracks = detect_tracks
    # get scores of lost tracks
    output_stracks = [
        track for track in self.tracked_stracks if track.is_activated
    ]
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


@TRACKER_REGISTRY.register()
class UTTracker2(UTTracker):

  def find_target(self, search_tensor, search_mask, scale_factor):
    """[summary]find tracking target and update informations

    Args:
        search_tensor ([type]): [description]
        search_mask ([type]): [description]
    Return:
      Box coordinate in xyxy format
    """
    with torch.no_grad():
      results = self.net.track_sot(search_tensor, search_mask,
                                   self.sot_ref_info)
    if results['pred_scores'] < 0.4:
      # pred_box = self.expand_search_box(results['pred_boxes'])[0]
      target_box = self.expand_search_box(self.target_box)
      # target_box = self.target_box
    else:
      pred_box = results['pred_boxes'][0]
      target_box = clip_box(self.map_box_back(pred_box, self.pos, scale_factor),
                            self.image_sz,
                            margin=10)
    self.pos = torch.Tensor([
        target_box[1] + (target_box[3] - 1) / 2,
        target_box[0] + (target_box[2] - 1) / 2
    ])
    self.target_sz = torch.Tensor([target_box[3], target_box[2]])
    return target_box

  def expand_search_box(self, box, scale=1.1):
    """use it when no target are find, expand search area 

    Args:
        boxes ([type]): [description]
        image_sizes ([type]): [description]

    Returns:
        [type]: [description]
    """
    w, h = box[2], box[3]
    new_w = min(min(self.image_sz) / 6.0, w * scale)
    new_h = min(min(self.image_sz) / 6.0, h * scale)
    l = box[0] - new_w / 2.0 + w / 2.0
    t = box[1] - new_h / 2.0 + h / 2.0
    return [l, t, new_w, new_h]

  def track_mot(self, image, info):

    resize_image = cv2.resize(image, self.resize[::-1])
    im_tensor = numpy_to_torch(resize_image).to(self.device) / 255.0
    mask_tensor = torch.zeros((1, *im_tensor.shape[-2:]),
                              dtype=torch.bool,
                              device=self.device)
    if self.use_pub_detection:
      public_detections = torch.tensor(info['init_detections']).to(self.device)
      public_detections[:, :4] = normalize_boxes(public_detections[:, :4],
                                                 image.shape[:2],
                                                 in_format='xywh',
                                                 out_format='cxcywh')
      self.mot_ref_info['public_detections'] = public_detections.unsqueeze(
          0)  ### batch dim
    # with torch.no_grad():
    # Predict the current location with KF
    if self.frame_num > 1 and len(self.last_detect_tracks) > 0:
      STrack.multi_predict(self.last_detect_tracks)
      for track in self.last_detect_tracks:
        track.update_trackbox(None)
      proposal = torch.tensor([t.tlwh for t in self.last_detect_tracks],
                              dtype=torch.float32).to(self.device)
      target_feature = torch.tensor(
          [t.curr_feat for t in self.last_detect_tracks],
          dtype=torch.float32).to(self.device)
      self.mot_ref_info['proposal'] = normalize_boxes(proposal,
                                                      image.shape[:2],
                                                      in_format='xywh',
                                                      out_format='cxcywh')[None]
      self.mot_ref_info['target_feat'] = target_feature[None]

    outputs, self.mot_ref_info = self.net.track_mot(im_tensor,
                                                    mask=mask_tensor,
                                                    ref_info=self.mot_ref_info)
    results = self.post_process(outputs, self.ori_image_szs, self.category)
    res_track = self.update_mot(results[0])

    # if self.visdom is not None:
    #   boxes = box_convert(results[0]['boxes'][results[0]['scores']>0.4], 'xyxy', 'xywh')
    #   self.visdom_draw_tracking(image, boxes)
    return res_track, results[0]

  def update_mot(self, output_results, use_embedding=False):
    scores = output_results["scores"].cpu().numpy()
    bboxes = output_results["boxes"].cpu().numpy()  # x1y1x2y2
    features = output_results["embs"].cpu().numpy()
    track_bboxes = output_results["track_boxes"].cpu().numpy(
    ) if "track_boxes" in output_results else None  # x1y1x2y2
    track_scores = output_results["track_scores"].cpu().numpy(
    ) if "track_scores" in output_results else None  # x1y1x2y2
    for idx, track in enumerate(self.last_detect_tracks):
      if track_scores[idx] > 0.8 and track.score > 0.95:
        track.update_trackbox(track_bboxes[idx])

    remain_inds = scores > self.score_thresh
    dets = bboxes[remain_inds]
    feats = features[remain_inds]
    scores_keep = scores[remain_inds]

    activated_starcks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []
    detect_tracks = []

    if len(dets) > 0:
      detections = [
          STrack(STrack.tlbr_to_tlwh(box), score, feat)
          for box, score, feat in zip(dets, scores_keep, feats)
      ]
    else:
      detections = []
    ''' Add newly detected stracks to tracked_stracks'''
    unconfirmed = []
    tracked_stracks = []
    for idx, track in enumerate(self.tracked_stracks):
      if not track.is_activated:
        unconfirmed.append(track)
      else:
        tracked_stracks.append(track)
    ''' Step 2: First association, with high score detection boxes'''
    # Predict the current location with KF
    STrack.multi_predict(self.lost_stracks)
    strack_pool = joint_tracklets(tracked_stracks, self.lost_stracks)
    # STrack.multi_predict(strack_pool)
    track_boxes = np.array([t.track_tlbr for t in strack_pool])
    det_boxes = np.array([d.tlbr for d in detections])
    dists = iou_distance(track_boxes, det_boxes)
    # dists = iou_distance(strack_pool, detections)
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
      detect_tracks.append(track)

    ## lost unmatched track
    for it in u_track:
      track = strack_pool[it]
      if not track.state == TrackState.Lost:
        track.mark_lost()
        lost_stracks.append(track)
    '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
    rest_detections = [detections[i] for i in u_detection]
    dists = iou_distance(unconfirmed, rest_detections)
    dists = fuse_score(dists, rest_detections)
    matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
    for itracked, idet in matches:
      unconfirmed[itracked].update(rest_detections[idet], self.frame_num)
      activated_starcks.append(unconfirmed[itracked])
      detect_tracks.append(unconfirmed[itracked])
    for it in u_unconfirmed:
      track = unconfirmed[it]
      track.mark_removed()
      removed_stracks.append(track)
    """ Step 4: Init new stracks"""
    for inew in u_detection:
      track = rest_detections[inew]
      if track.score < self.score_thresh:
        continue
      track.activate(self.kalman_filter, self.frame_num)
      activated_starcks.append(track)
      detect_tracks.append(track)
    """ Step 5: Update state"""
    for track in self.lost_stracks:
      if self.frame_num - track.end_frame > self.max_age:
        track.mark_removed()
        removed_stracks.append(track)

    # print('Ramained match {} s'.format(t4-t3))

    self.tracked_stracks = [
        t for t in self.tracked_stracks if t.state == TrackState.Tracked
    ]
    self.tracked_stracks = joint_tracklets(self.tracked_stracks,
                                         activated_starcks)
    self.tracked_stracks = joint_tracklets(self.tracked_stracks, refind_stracks)
    self.lost_stracks = sub_tracklets(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = sub_tracklets(self.lost_stracks, self.removed_stracks)
    self.removed_stracks.extend(removed_stracks)
    self.tracked_stracks, self.lost_stracks = remove_duplicate_tracklets(
        self.tracked_stracks, self.lost_stracks)
    # self.last_detect_tracks = detect_tracks
    self.last_detect_tracks = [
        track for track in self.tracked_stracks if track.is_activated
    ]
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
            # 'bbox': track.track_tlbr,
            'score': track.score,
            'active': 1.0,
        }
        output_stracks.append(track_obj)

    return output_stracks

