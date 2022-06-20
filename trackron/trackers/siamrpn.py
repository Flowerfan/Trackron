# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from trackron.structures import Anchors

from .base_tracker import BaseTracker
from .build import TRACKER_REGISTRY
from .parameters import get_params


@TRACKER_REGISTRY.register()
class SiamRPNTracker(BaseTracker):

  def setup(self):
    assert self.tracking_mode == 'sot', 'siamrpn only support SOT nonly'
    self.params = get_params(self.cfg.TRACKER.PARAMETER)
    self.output_score = self.cfg.TRACKER.OUTPUT_SCORE
    hanning = np.hanning(self.params.score_size)
    window = np.outer(hanning, hanning)
    self.window = np.tile(window.flatten(), self.params.anchor_nums)
    self.anchors = self.generate_anchor(self.params.score_size)

  def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
    """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
    if isinstance(pos, float):
      pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
      size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
      te_im = np.zeros(size, np.uint8)
      te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
      if top_pad:
        te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
      if bottom_pad:
        te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
      if left_pad:
        te_im[:, 0:left_pad, :] = avg_chans
      if right_pad:
        te_im[:, c + left_pad:, :] = avg_chans
      im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
    else:
      im_patch = im[int(context_ymin):int(context_ymax + 1),
                    int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
      im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)
    im_patch = torch.from_numpy(im_patch)
    return im_patch


  def generate_anchor(self, score_size):
    anchors = Anchors(self.params.stride, self.params.ratios, self.params.scales)
    anchor = anchors.anchors
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
    total_stride = anchors.stride
    anchor_num = anchor.shape[0]
    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = -(score_size // 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
        np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor

  def _convert_bbox(self, delta, anchor):
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
    delta = delta.data.cpu().numpy()

    delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
    delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
    return delta

  def _convert_score(self, score):
    score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
    score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
    return score

  def _bbox_clip(self, cx, cy, width, height, boundary):
    cx = max(0, min(cx, boundary[1]))
    cy = max(0, min(cy, boundary[0]))
    width = max(10, min(width, boundary[1]))
    height = max(10, min(height, boundary[0]))
    return cx, cy, width, height

  def init_sot(self, image, info: dict, **kwargs) -> dict:
    """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tic = time.time()
    bbox = info['init_bbox']
    self.center_pos = np.array(
        [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
    self.size = np.array([bbox[2], bbox[3]])

    # calculate z crop size
    w_z = self.size[0] + self.params.context_amount * np.sum(self.size)
    h_z = self.size[1] + self.params.context_amount * np.sum(self.size)
    s_z = round(np.sqrt(w_z * h_z))

    # calculate channle average
    self.channel_average = np.mean(image, axis=(0, 1))

    # get crop
    z_crop = self.get_subwindow(image, self.center_pos, self.params.exemplar_size,
                                s_z, self.channel_average)
    self.sot_ref_info = self.net.track_sot(z_crop)
    out = {'time': time.time() - tic}
    out['target_bbox'] = bbox
    if self.output_score:
      out['target_bbox'] = self.target_box + [1.0]
    return out

  def track_sot(self, img, info: dict = None):
    """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    w_z = self.size[0] + self.params.context_amount * np.sum(self.size)
    h_z = self.size[1] + self.params.context_amount * np.sum(self.size)
    s_z = np.sqrt(w_z * h_z)
    scale_z = self.params.exemplar_size / s_z
    s_x = s_z * (self.params.instance_size / self.params.exemplar_size)
    x_crop = self.get_subwindow(img, self.center_pos, self.params.instance_size,
                                round(s_x), self.channel_average)

    outputs = self.net.track_sot(x_crop, ref_info=self.sot_ref_info)

    score = self._convert_score(outputs['cls'])
    pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

    def change(r):
      return np.maximum(r, 1. / r)

    def sz(w, h):
      pad = (w + h) * 0.5
      return np.sqrt((w + pad) * (h + pad))

    # scale penalty
    s_c = change(
        sz(pred_bbox[2, :], pred_bbox[3, :]) /
        (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

    # aspect ratio penalty
    r_c = change(
        (self.size[0] / self.size[1]) / (pred_bbox[2, :] / pred_bbox[3, :]))
    penalty = np.exp(-(r_c * s_c - 1) * self.params.penalty_k)
    pscore = penalty * score

    # window penalty
    pscore = pscore * (1 - self.params.window_influence) + \
        self.window * self.params.window_influence
    best_idx = np.argmax(pscore)

    bbox = pred_bbox[:, best_idx] / scale_z
    lr = penalty[best_idx] * score[best_idx] * self.params.lr

    cx = bbox[0] + self.center_pos[0]
    cy = bbox[1] + self.center_pos[1]

    # smooth bbox
    width = self.size[0] * (1 - lr) + bbox[2] * lr
    height = self.size[1] * (1 - lr) + bbox[3] * lr

    # clip boundary
    cx, cy, width, height = self._bbox_clip(cx, cy, width, height,
                                            img.shape[:2])

    # udpate state
    self.center_pos = np.array([cx, cy])
    self.size = np.array([width, height])

    bbox = [cx - width / 2, cy - height / 2, width, height]
    if self.output_score:
      bbox = bbox + [score[best_idx]]
    out = {'target_bbox': bbox}
    return out
