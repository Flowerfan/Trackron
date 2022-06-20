import copy
import time

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
from trackron.data.utils import (get_size_with_aspect_ratio, normalize_boxes,
                                 sample_target_brpadding)

from .base_tracker import BaseTracker, PostProcess, PublicPostProcess, clip_box
from .build import TRACKER_REGISTRY
from .utils.preprocessing import numpy_to_torch, sample_target


def get_threshold(video_name):
  thresh = {
      'MOT16-01': 0.55,
      'MOT16-06': 0.55,
      'MOT16-12': 0.6,
      'MOT16-14': 0.6
  }
  return thresh.get(video_name, 0.4)


def get_max_age(video_name):
  max_age = {'MOT16-05': 14, 'MOT16-06': 14, 'MOT16-12': 25, 'MOT16-14': 25}
  return max_age.get(video_name, 30)


@TRACKER_REGISTRY.register()
class SiameseTracker(BaseTracker):

  def setup(self):
    if self.tracking_mode != 'mot':
      self.image_sample_size = self.cfg.SOT.DATASET.SEARCH.SIZE
      self.search_area_scale = self.cfg.SOT.DATASET.SEARCH.FACTOR
    self.sample_memory_size = self.cfg.TRACKER.MEMORY_SIZE
    self.output_score = self.cfg.TRACKER.OUTPUT_SCORE
    self.use_pub_detection = self.cfg.TRACKER.PUBLIC_DETECTION
    self.det_thresh = self.cfg.TRACKER.DETECTION_THRESH
    self.track_thresh = self.cfg.TRACKER.TRACKING_THRESH
    self.match_thresh = self.cfg.TRACKER.MATCHING_THRESH

  def init_sot(self, image, info: dict, **kwargs) -> dict:

    # Time initialization
    tic = time.time()

    # Get object id
    self.object_id = info.get('object_ids', [None])

    # Get target position and size
    state = info['init_bbox']
    self.target_box = state
    self.pos = torch.Tensor(
        [state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
    self.target_sz = torch.Tensor([state[3], state[2]])
    self.image_sz = image.shape[:2]

    # Crop Image
    template_arr, scale_factor, template_mask_arr = sample_target(
        image,
        info['init_bbox'],
        self.search_area_scale,
        output_sz=self.image_sample_size)
    template_tensor = numpy_to_torch(template_arr).to(self.device) / 255.0
    template_mask = torch.from_numpy(template_mask_arr).to(torch.bool).to(
        self.device)[None]
    template_box = self.get_cropped_img_box(self.pos, self.target_sz,
                                            self.pos, scale_factor).reshape(
                                                1, 4)  ## xyxy format

    self.init_memory(template_tensor, template_box, template_mask)

    out = {'time': time.time() - tic}
    out['target_bbox'] = self.target_box
    if self.output_score:
      out['target_bbox'] = self.target_box + [1.0]
    return out

  def init_memory(self, tensor, box, mask):
    ### initialize first frame
    with torch.no_grad():
      self.sot_ref_info = self.net.track_sot(tensor, mask, init_box=box)

  def track_sot(self, image, info: dict = None) -> dict:

    # prepare search image
    img_arr, scale_factor, img_mask = sample_target(
        image,
        self.target_box,
        self.search_area_scale,
        output_sz=self.image_sample_size)

    ### update target box coord in cropped image
    proposal_box = self.get_cropped_img_box(self.pos, self.target_sz,
                                            self.pos, scale_factor).reshape(
                                                1, 4)  ## xyxy format
    self.sot_ref_info['target_boxes'] = proposal_box.to(
        self.device) / self.image_sample_size
    self.sot_ref_info['image_sizes'] = img_arr.shape[:2]

    # prepare inputs
    search_tensor = numpy_to_torch(img_arr).to(self.device) / 255.0
    search_mask = torch.from_numpy(img_mask).to(torch.bool).to(
        self.device)[None]
    # search_mask = torch.zeros((1, *search_tensor.shape[-2:]),
    #                           device=self.device,
    #                           dtype=torch.bool)

    self.target_box = self.find_target(search_tensor, search_mask, scale_factor)
    # pred_box = (pred_box / scale_factor).tolist()  # (x1, y1, x2, y2)

    output_state = self.target_box

    if self.output_score:
      output_state = output_state + [1.0]

    out = {'target_bbox': output_state}
    return out

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
    if results.get('pred_scores', 1.0) < -10:
      # pred_box = self.expand_search_box(results['pred_boxes'])[0]
      target_box = self.expand_search_box(self.target_box)
      # target_box = self.target_box
    else:
      pred_box = results['pred_boxes'][0]
      target_box = clip_box(self.map_box_back(pred_box, self.pos, scale_factor),
                            self.image_sz,
                            margin=2)
    self.pos = torch.Tensor([
        target_box[1] + (target_box[3] - 1) / 2,
        target_box[0] + (target_box[2] - 1) / 2
    ])
    self.target_sz = torch.Tensor([target_box[3], target_box[2]])
    return target_box

  # def map_box_back(self, pred_box: list, scale_factor: float):
  #   """xyxy in resized image to xywh in origin image"""
  #   cx_prev, cy_prev = self.target_box[0] + 0.5 * self.target_box[
  #       2], self.target_box[1] + 0.5 * self.target_box[3]
  #   x1, y1, x2, y2 = pred_box
  #   half_side = 0.5 * self.image_sample_size / scale_factor
  #   x_real = x1 + (cx_prev - half_side)
  #   y_real = y1 + (cy_prev - half_side)
  #   return [x_real, y_real, x2 - x1, y2 - y1]

  def map_box_back(self, pred_box: list, image_center: list,
                   scale_factor: float):
    """xyxy in resized image to xywh in origin image"""
    x1, y1, x2, y2 = (pred_box / scale_factor).tolist()
    tl_y, tl_x = (image_center -
                  0.5 * self.image_sample_size / scale_factor).tolist()
    x_real = x1 + tl_x
    y_real = y1 + tl_y
    return [x_real, y_real, x2 - x1, y2 - y1]

  def init_vos(self, image, info: dict, **kwargs) -> dict:
    # Time initialization
    tic = time.time()

    # Get object id
    self.object_id = info.get('object_ids', [None])

    # Get target position and size
    init_boxes = list(info['init_bbox'].values())
    # init_masks = list(info['init_mask'].values())

    template_tensor = numpy_to_torch(image).to(self.device) / 255.0
    template_bbox = torch.tensor(init_boxes,
                                 dtype=torch.float32,
                                 device=self.device)[None]  ### B N 4
    template_mask = torch.zeros_like(template_tensor)[:, 0]  ### B H W

    with torch.no_grad():
      self.ref_info = self.net.track_sot(template_tensor,
                                         template_mask,
                                         init_box=template_bbox)

    out = {'time': time.time() - tic}
    out['segmentation'] = info['init_mask']
    return out

  def track_vos(self, image, info: dict = None) -> dict:
    H, W = image.shape[:2]
    im_tensor = numpy_to_torch(image).to(self.device) / 255.0
    mask_tensor = torch.zeros((1, *im_tensor.shape[-2:]),
                              dtype=torch.bool,
                              device=self.device)

    with torch.no_grad():
      results = self.net.track_sot(im_tensor, mask_tensor, self.ref_info)
    out_mask = np.zeros((H, W), dtype=np.uint8)
    for idx, pred_mask in enumerate(results['pred_masks']):
      out_mask[pred_mask.cpu().numpy()] = idx + 1
    out = {'segmentation': out_mask}
    return out

  def reset_mot(self):
    self.id_count = 0
    self.tracks_dict = dict()
    self.tracks = list()
    self.unmatched_tracks = list()

  def init_mot(self, image, info: dict, category=None) -> dict:

    # self.score_thresh = get_threshold(info['video'])
    # self.max_age = get_max_age(info['video'])
    self.score_thresh = self.det_thresh
    self.max_age = 10
    self.id_count = 0
    self.id_color = None
    self.tracks_dict = dict()
    self.tracks = list()
    self.unmatched_tracks = list()
    self.reset_mot()
    self.category = category

    # Time initialization
    tic = time.time()
    self.resize = get_size_with_aspect_ratio(image.shape[:2])
    # self.resize = (image.shape[1], image.shape[0])
    # self.resize = (800, 1440)
    # self.resize = (608, 1088)
    self.ori_image_szs = torch.tensor([image.shape[:2]],
                                      dtype=torch.float32,
                                      device=self.device)
    resize_image = cv2.resize(image, self.resize[::-1])
    if self.use_pub_detection:
      self.post_process = PublicPostProcess()
      public_detections = torch.tensor(info['init_detections']).to(self.device)
      public_detections[:, :4] = normalize_boxes(public_detections[:, :4],
                                                 image.shape[:2],
                                                 in_format='xywh',
                                                 out_format='cxcywh')
      self.mot_ref_info = {
          'public_detections':
              public_detections.unsqueeze(0)  ### batch dim
      }
    else:
      self.post_process = PostProcess()
      self.mot_ref_info = {}

    # Crop Image
    im_tensor = numpy_to_torch(resize_image).to(self.device) / 255.0
    self.mask_tensor = torch.zeros((1, *im_tensor.shape[-2:]),
                                   dtype=torch.bool,
                                   device=self.device)
    with torch.no_grad():
      outputs, mot_ref_info = self.net.track_mot(im_tensor,
                                                 mask=self.mask_tensor,
                                                 ref_info=self.mot_ref_info)
    self.mot_ref_info = mot_ref_info

    results = self.post_process(outputs, self.ori_image_szs, category)
    res_track = self.init_mot_memory(results[0])

    # self.visdom = Visdom(debug=2)
    # boxes = box_convert(results[0]['boxes'][results[0]['scores']>0.4], 'xyxy', 'xywh')
    # self.visdom_draw_tracking(image, boxes)
    return res_track, results[0]

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
    with torch.no_grad():
      outputs, mot_ref_info = self.net.track_mot(im_tensor,
                                                 mask=mask_tensor,
                                                 ref_info=self.mot_ref_info)
    self.mot_ref_info = mot_ref_info
    results = self.post_process(outputs, self.ori_image_szs, self.category)
    res_track = self.update_mot_memory(results[0])

    # if self.visdom is not None:
    #   boxes = box_convert(results[0]['boxes'][results[0]['scores']>0.4], 'xyxy', 'xywh')
    #   self.visdom_draw_tracking(image, boxes)
    return res_track, results[0]

  def init_mot_memory(self, results):

    scores = results["scores"]
    classes = results["labels"]
    bboxes = results["boxes"]  # x1y1x2y2
    features = results['embs']

    ret = list()
    ret_dict = dict()
    for idx in range(scores.shape[0]):
      if scores[idx] >= self.score_thresh:
        self.id_count += 1
        obj = dict()
        obj["score"] = float(scores[idx])
        obj['class'] = int(classes[idx])
        obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
        # obj['feature'] = features[idx]
        obj["tracking_id"] = self.id_count
        #                 obj['vxvy'] = [0.0, 0.0]
        obj['active'] = 1
        obj['age'] = 1
        ret.append(obj)
        ret_dict[idx] = obj

    self.tracks = ret
    self.tracks_dict = ret_dict
    return copy.deepcopy(ret)

  def update_mot_memory(self, output_results, use_embedding=False):
    scores = output_results["scores"]
    classes = output_results["labels"]
    bboxes = output_results["boxes"]  # x1y1x2y2
    if use_embedding:
      features = output_results["features"]
    track_bboxes = output_results[
        "track_boxes"] if "track_boxes" in output_results else None  # x1y1x2y2
    track_scores = output_results[
        "track_scores"] if "track_scores" in output_results else None  # x1y1x2y2

    results = list()
    detection_dict = dict()

    tracks = list()

    ### udpate tracking results
    for idx in range(track_bboxes.shape[0]):
      if idx in self.tracks_dict and track_bboxes is not None:
        if track_scores is None:
          self.tracks_dict[idx]["bbox"] = track_bboxes[
              idx, :].cpu().numpy().tolist()
        elif track_scores[idx] > self.track_thresh:
          self.tracks_dict[idx]["bbox"] = track_bboxes[
              idx, :].cpu().numpy().tolist()

    ### update detection results
    for idx in range(scores.shape[0]):
      if scores[idx] >= self.score_thresh:
        obj = dict()
        obj["score"] = float(scores[idx])
        obj['class'] = int(classes[idx])
        obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
        results.append(obj)
        detection_dict[idx] = obj

    tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
    N = len(results)
    M = len(tracks)

    ret = list()
    unmatched_tracks = [t for t in range(M)]
    unmatched_dets = [d for d in range(N)]
    if N > 0 and M > 0:
      det_box = torch.stack([torch.tensor(obj['bbox']) for obj in results],
                            dim=0)  # N x 4
      track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks],
                              dim=0)  # M x 4
      cost_bbox = 1.0 - generalized_box_iou(det_box,
                                            track_box).cpu().numpy()  # N x M
      # cost_bbox = 1.0 - ious(det_box.cpu().numpy(), track_box.cpu().numpy())

      matched_indices = linear_sum_assignment(cost_bbox)
      unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
      unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]
      # matched_indices, unmatched_dets, unmatched_tracks = linear_assignment(cost_bbox, 1.0)

      matches = [[], []]
      for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
        if cost_bbox[m0, m1] > self.match_thresh:
          unmatched_dets.append(m0)
          unmatched_tracks.append(m1)
        else:
          matches[0].append(m0)
          matches[1].append(m1)

      for (m0, m1) in zip(matches[0], matches[1]):
        # for (m0, m1) in matched_indices:
        track = results[m0]
        track['tracking_id'] = tracks[m1]['tracking_id']
        track['age'] = 1
        track['active'] = 1
        # track['bbox'] = tracks[m1]['bbox']
        pre_box = tracks[m1]['bbox']
        cur_box = track['bbox']
        #             pre_cx, pre_cy = (pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2
        #             cur_cx, cur_cy = (cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2
        #             track['vxvy'] = [cur_cx - pre_cx, cur_cy - pre_cy]
        ret.append(track)

    for i in unmatched_dets:
      track = results[i]
      self.id_count += 1
      track['tracking_id'] = self.id_count
      track['age'] = 1
      track['active'] = 1
      #             track['vxvy'] = [0.0, 0.0]
      ret.append(track)

    ret_unmatched_tracks = []
    for i in unmatched_tracks:
      track = tracks[i]
      if track['age'] < self.max_age:
        track['age'] += 1
        track['active'] = 0
        #                 x1, y1, x2, y2 = track['bbox']
        #                 vx, vy = track['vxvy']
        #                 track['bbox'] = [x1+vx, y1+vy, x2+vx, y2+vy]
        ret.append(track)
        ret_unmatched_tracks.append(track)

    self.tracks = ret
    self.tracks_dict = detection_dict
    self.unmatched_tracks = ret_unmatched_tracks
    return copy.deepcopy(ret)

  def get_cropped_img_box(self, pos, sz, sample_pos, sample_scale):
    """All inputs in original image coordinates.
    Generates a box in the cropped image sample reference frame, in the xyxy format used by the model."""
    box_center = (pos - sample_pos) * sample_scale + (self.image_sample_size -
                                                      1) / 2
    box_sz = sz * sample_scale
    target_ul = box_center - (box_sz - 1) / 2
    target_br = box_center + 1 + (box_sz - 1) / 2
    return torch.cat([target_ul.flip((0,)), target_br.flip((0,))])

  def visualize_score_map(self, score, name='top_left_score'):
    self.visdom.register(score, 'heatmap', 2, name)

  def visualize_search(self, image, box=None):
    ###box xyxy format
    if box is not None:
      box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
      self.visdom.register((image, box), 'Tracking', 1, 'Search')
    else:
      self.visdom.register(image, 'image', 1, 'init')
