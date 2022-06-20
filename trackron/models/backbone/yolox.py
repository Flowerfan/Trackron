#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert
from trackron.data.utils import iou
from trackron.trackers.utils.postprocessing import postprocess


from .darknet import BaseConv, CSPDarknet, CSPLayer, Darknet, DWConv
from .build import BACKBONE_REGISTRY


class IOUloss(nn.Module):

  def __init__(self, reduction="none", loss_type="iou"):
    super(IOUloss, self).__init__()
    self.reduction = reduction
    self.loss_type = loss_type

  def forward(self, pred, target):
    assert pred.shape[0] == target.shape[0]

    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    tl = torch.max((pred[:, :2] - pred[:, 2:] / 2),
                   (target[:, :2] - target[:, 2:] / 2))
    br = torch.min((pred[:, :2] + pred[:, 2:] / 2),
                   (target[:, :2] + target[:, 2:] / 2))

    area_p = torch.prod(pred[:, 2:], 1)
    area_g = torch.prod(target[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    iou = (area_i) / (area_p + area_g - area_i + 1e-16)

    if self.loss_type == "iou":
      loss = 1 - iou**2
    elif self.loss_type == "giou":
      c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                       (target[:, :2] - target[:, 2:] / 2))
      c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                       (target[:, :2] + target[:, 2:] / 2))
      area_c = torch.prod(c_br - c_tl, 1)
      giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
      loss = 1 - giou.clamp(min=-1.0, max=1.0)

    if self.reduction == "mean":
      loss = loss.mean()
    elif self.reduction == "sum":
      loss = loss.sum()

    return loss


class YOLOFPN(nn.Module):
  """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

  def __init__(
      self,
      depth=53,
      in_features=["dark3", "dark4", "dark5"],
  ):
    super().__init__()

    self.backbone = Darknet(depth)
    self.in_features = in_features

    # out 1
    self.out1_cbl = self._make_cbl(512, 256, 1)
    self.out1 = self._make_embedding([256, 512], 512 + 256)

    # out 2
    self.out2_cbl = self._make_cbl(256, 128, 1)
    self.out2 = self._make_embedding([128, 256], 256 + 128)

    # upsample
    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

  def _make_cbl(self, _in, _out, ks):
    return BaseConv(_in, _out, ks, stride=1, act="lrelu")

  def _make_embedding(self, filters_list, in_filters):
    m = nn.Sequential(*[
        self._make_cbl(in_filters, filters_list[0], 1),
        self._make_cbl(filters_list[0], filters_list[1], 3),
        self._make_cbl(filters_list[1], filters_list[0], 1),
        self._make_cbl(filters_list[0], filters_list[1], 3),
        self._make_cbl(filters_list[1], filters_list[0], 1),
    ])
    return m

  def load_pretrained_model(self, filename="./weights/darknet53.mix.pth"):
    with open(filename, "rb") as f:
      state_dict = torch.load(f, map_location="cpu")
    print("loading pretrained weights...")
    self.backbone.load_state_dict(state_dict)

  def forward(self, inputs):
    """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
    #  backbone
    out_features = self.backbone(inputs)
    x2, x1, x0 = [out_features[f] for f in self.in_features]

    #  yolo branch 1
    x1_in = self.out1_cbl(x0)
    x1_in = self.upsample(x1_in)
    x1_in = torch.cat([x1_in, x1], 1)
    out_dark4 = self.out1(x1_in)

    #  yolo branch 2
    x2_in = self.out2_cbl(out_dark4)
    x2_in = self.upsample(x2_in)
    x2_in = torch.cat([x2_in, x2], 1)
    out_dark3 = self.out2(x2_in)

    outputs = (out_dark3, out_dark4, x0)
    return outputs


class YOLOPAFPN(nn.Module):
  """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

  def __init__(
      self,
      depth=1.0,
      width=1.0,
      in_features=("dark3", "dark4", "dark5"),
      in_channels=[256, 512, 1024],
      depthwise=False,
      act="silu",
  ):
    super().__init__()
    self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
    self.in_features = in_features
    self.in_channels = in_channels
    Conv = DWConv if depthwise else BaseConv

    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    self.lateral_conv0 = BaseConv(int(in_channels[2] * width),
                                  int(in_channels[1] * width),
                                  1,
                                  1,
                                  act=act)
    self.C3_p4 = CSPLayer(
        int(2 * in_channels[1] * width),
        int(in_channels[1] * width),
        round(3 * depth),
        False,
        depthwise=depthwise,
        act=act,
    )  # cat

    self.reduce_conv1 = BaseConv(int(in_channels[1] * width),
                                 int(in_channels[0] * width),
                                 1,
                                 1,
                                 act=act)
    self.C3_p3 = CSPLayer(
        int(2 * in_channels[0] * width),
        int(in_channels[0] * width),
        round(3 * depth),
        False,
        depthwise=depthwise,
        act=act,
    )

    # bottom-up conv
    self.bu_conv2 = Conv(int(in_channels[0] * width),
                         int(in_channels[0] * width),
                         3,
                         2,
                         act=act)
    self.C3_n3 = CSPLayer(
        int(2 * in_channels[0] * width),
        int(in_channels[1] * width),
        round(3 * depth),
        False,
        depthwise=depthwise,
        act=act,
    )

    # bottom-up conv
    self.bu_conv1 = Conv(int(in_channels[1] * width),
                         int(in_channels[1] * width),
                         3,
                         2,
                         act=act)
    self.C3_n4 = CSPLayer(
        int(2 * in_channels[1] * width),
        int(in_channels[2] * width),
        round(3 * depth),
        False,
        depthwise=depthwise,
        act=act,
    )

  def forward(self, input):
    """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

    #  backbone
    out_features = self.backbone(input)
    features = [out_features[f] for f in self.in_features]
    [x2, x1, x0] = features

    fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
    f_out0 = self.upsample(fpn_out0)  # 512/16
    f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
    f_out0 = self.C3_p4(f_out0)  # 1024->512/16

    fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
    f_out1 = self.upsample(fpn_out1)  # 256/8
    f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
    pan_out2 = self.C3_p3(f_out1)  # 512->256/8

    p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
    p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
    pan_out1 = self.C3_n3(p_out1)  # 512->512/16

    p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
    p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
    pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

    outputs = (pan_out2, pan_out1, pan_out0)
    return outputs


class YOLOXHead(nn.Module):

  def __init__(
      self,
      num_classes,
      width=1.0,
      strides=[8, 16, 32],
      in_channels=[256, 512, 1024],
      act="silu",
      depthwise=False,
      conf_thresh=0.7,
      nms_thresh=0.45,
  ):
    """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
        """
    super().__init__()

    self.n_anchors = 1
    self.num_classes = num_classes
    self.decode_in_inference = True  # for deploy, set to False
    self.conf_thresh = conf_thresh
    self.nms_thresh = nms_thresh

    self.cls_convs = nn.ModuleList()
    self.reg_convs = nn.ModuleList()
    self.cls_preds = nn.ModuleList()
    self.reg_preds = nn.ModuleList()
    self.obj_preds = nn.ModuleList()
    self.stems = nn.ModuleList()
    Conv = DWConv if depthwise else BaseConv

    for i in range(len(in_channels)):
      self.stems.append(
          BaseConv(
              in_channels=int(in_channels[i] * width),
              out_channels=int(256 * width),
              ksize=1,
              stride=1,
              act=act,
          ))
      self.cls_convs.append(
          nn.Sequential(*[
              Conv(
                  in_channels=int(256 * width),
                  out_channels=int(256 * width),
                  ksize=3,
                  stride=1,
                  act=act,
              ),
              Conv(
                  in_channels=int(256 * width),
                  out_channels=int(256 * width),
                  ksize=3,
                  stride=1,
                  act=act,
              ),
          ]))
      self.reg_convs.append(
          nn.Sequential(*[
              Conv(
                  in_channels=int(256 * width),
                  out_channels=int(256 * width),
                  ksize=3,
                  stride=1,
                  act=act,
              ),
              Conv(
                  in_channels=int(256 * width),
                  out_channels=int(256 * width),
                  ksize=3,
                  stride=1,
                  act=act,
              ),
          ]))
      self.cls_preds.append(
          nn.Conv2d(
              in_channels=int(256 * width),
              out_channels=self.n_anchors * self.num_classes,
              kernel_size=1,
              stride=1,
              padding=0,
          ))
      self.reg_preds.append(
          nn.Conv2d(
              in_channels=int(256 * width),
              out_channels=4,
              kernel_size=1,
              stride=1,
              padding=0,
          ))
      self.obj_preds.append(
          nn.Conv2d(
              in_channels=int(256 * width),
              out_channels=self.n_anchors * 1,
              kernel_size=1,
              stride=1,
              padding=0,
          ))

    self.use_l1 = False
    self.l1_loss = nn.L1Loss(reduction="none")
    self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
    self.iou_loss = IOUloss(reduction="none")
    self.strides = strides
    self.grids = [torch.zeros(1)] * len(in_channels)
    self.expanded_strides = [None] * len(in_channels)

  def initialize_biases(self, prior_prob):
    for conv in self.cls_preds:
      b = conv.bias.view(self.n_anchors, -1)
      b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
      conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    for conv in self.obj_preds:
      b = conv.bias.view(self.n_anchors, -1)
      b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
      conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

  def forward(self, xin, labels=None, imgs=None):
    outputs = []
    origin_preds = []
    x_shifts = []
    y_shifts = []
    expanded_strides = []

    for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
        zip(self.cls_convs, self.reg_convs, self.strides, xin)):
      x = self.stems[k](x)
      cls_x = x
      reg_x = x

      cls_feat = cls_conv(cls_x)
      cls_output = self.cls_preds[k](cls_feat)

      reg_feat = reg_conv(reg_x)
      reg_output = self.reg_preds[k](reg_feat)
      obj_output = self.obj_preds[k](reg_feat)

      if self.training:
        output = torch.cat([reg_output, obj_output, cls_output], 1)
        output, grid = self.get_output_and_grid(output, k, stride_this_level,
                                                xin[0].type())
        x_shifts.append(grid[:, :, 0])
        y_shifts.append(grid[:, :, 1])
        expanded_strides.append(
            torch.zeros(1,
                        grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
        if self.use_l1:
          batch_size = reg_output.shape[0]
          hsize, wsize = reg_output.shape[-2:]
          reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize,
                                       wsize)
          reg_output = reg_output.permute(0, 1, 3, 4,
                                          2).reshape(batch_size, -1, 4)
          origin_preds.append(reg_output.clone())

      else:
        output = torch.cat(
            [reg_output, obj_output.sigmoid(),
             cls_output.sigmoid()], 1)

      outputs.append(output)

    if self.training:
      return self.get_losses(
          imgs,
          x_shifts,
          y_shifts,
          expanded_strides,
          labels,
          torch.cat(outputs, 1),
          origin_preds,
          dtype=xin[0].dtype,
      )
    else:
      self.hw = [x.shape[-2:] for x in outputs]
      # [batch, n_anchors_all, 85]
      outputs = torch.cat([x.flatten(start_dim=2) for x in outputs],
                          dim=2).permute(0, 2, 1)
      if self.decode_in_inference:
        return self.decode_outputs(outputs, dtype=xin[0].type())
      else:
        return outputs

  def get_output_and_grid(self, output, k, stride, dtype):
    grid = self.grids[k]

    batch_size = output.shape[0]
    n_ch = 5 + self.num_classes
    hsize, wsize = output.shape[-2:]
    if grid.shape[2:4] != output.shape[2:4]:
      yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
      grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
      self.grids[k] = grid

    output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
    output = output.permute(0, 1, 3, 4,
                            2).reshape(batch_size,
                                       self.n_anchors * hsize * wsize, -1)
    grid = grid.view(1, -1, 2)
    output[..., :2] = (output[..., :2] + grid) * stride
    output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
    return output, grid

  def decode_outputs(self, outputs, dtype):
    grids = []
    strides = []
    for (hsize, wsize), stride in zip(self.hw, self.strides):
      yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
      grid = torch.stack((xv, yv), 2).view(1, -1, 2)
      grids.append(grid)
      shape = grid.shape[:2]
      strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    strides = torch.cat(strides, dim=1).type(dtype)

    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

    outputs = postprocess(outputs, self.num_classes, conf_thresh=self.conf_thresh, nms_thresh=self.nms_thresh)

    results = {'pred_boxes': outputs[..., :4],
               'object_scores': outputs[..., 4],
               'class_scores': outputs[..., 5],
               'pred_classes': outputs[..., 6].long()
               }
    return results
  
  


  def get_losses(
      self,
      imgs,
      x_shifts,
      y_shifts,
      expanded_strides,
      labels,
      outputs,
      origin_preds,
      dtype,
  ):
    bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
    obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
    cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

    # calculate targets
    mixup = labels.shape[2] > 5
    if mixup:
      label_cut = labels[..., :5]
    else:
      label_cut = labels
    nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

    total_num_anchors = outputs.shape[1]
    x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
    y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
    expanded_strides = torch.cat(expanded_strides, 1)
    if self.use_l1:
      origin_preds = torch.cat(origin_preds, 1)

    cls_targets = []
    reg_targets = []
    l1_targets = []
    obj_targets = []
    fg_masks = []

    num_fg = 0.0
    num_gts = 0.0

    for batch_idx in range(outputs.shape[0]):
      num_gt = int(nlabel[batch_idx])
      num_gts += num_gt
      if num_gt == 0:
        cls_target = outputs.new_zeros((0, self.num_classes))
        reg_target = outputs.new_zeros((0, 4))
        l1_target = outputs.new_zeros((0, 4))
        obj_target = outputs.new_zeros((total_num_anchors, 1))
        fg_mask = outputs.new_zeros(total_num_anchors).bool()
      else:
        gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
        gt_classes = labels[batch_idx, :num_gt, 0]
        bboxes_preds_per_image = bbox_preds[batch_idx]

        try:
          (
              gt_matched_classes,
              fg_mask,
              pred_ious_this_matching,
              matched_gt_inds,
              num_fg_img,
          ) = self.get_assignments(  # noqa
              batch_idx,
              num_gt,
              total_num_anchors,
              gt_bboxes_per_image,
              gt_classes,
              bboxes_preds_per_image,
              expanded_strides,
              x_shifts,
              y_shifts,
              cls_preds,
              bbox_preds,
              obj_preds,
              labels,
              imgs,
          )
        except RuntimeError:
          logger = logging.getLogger(__name__)
          logger.info(
              "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size.")
          print(
              "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size.")
          torch.cuda.empty_cache()
          (
              gt_matched_classes,
              fg_mask,
              pred_ious_this_matching,
              matched_gt_inds,
              num_fg_img,
          ) = self.get_assignments(  # noqa
              batch_idx,
              num_gt,
              total_num_anchors,
              gt_bboxes_per_image,
              gt_classes,
              bboxes_preds_per_image,
              expanded_strides,
              x_shifts,
              y_shifts,
              cls_preds,
              bbox_preds,
              obj_preds,
              labels,
              imgs,
              "cpu",
          )

        torch.cuda.empty_cache()
        num_fg += num_fg_img

        cls_target = F.one_hot(
            gt_matched_classes.to(torch.int64),
            self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
        obj_target = fg_mask.unsqueeze(-1)
        reg_target = gt_bboxes_per_image[matched_gt_inds]

        if self.use_l1:
          l1_target = self.get_l1_target(
              outputs.new_zeros((num_fg_img, 4)),
              gt_bboxes_per_image[matched_gt_inds],
              expanded_strides[0][fg_mask],
              x_shifts=x_shifts[0][fg_mask],
              y_shifts=y_shifts[0][fg_mask],
          )

      cls_targets.append(cls_target)
      reg_targets.append(reg_target)
      obj_targets.append(obj_target.to(dtype))
      fg_masks.append(fg_mask)
      if self.use_l1:
        l1_targets.append(l1_target)

    cls_targets = torch.cat(cls_targets, 0)
    reg_targets = torch.cat(reg_targets, 0)
    obj_targets = torch.cat(obj_targets, 0)
    fg_masks = torch.cat(fg_masks, 0)
    if self.use_l1:
      l1_targets = torch.cat(l1_targets, 0)

    num_fg = max(num_fg, 1)
    loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks],
                              reg_targets)).sum() / num_fg
    loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1),
                                     obj_targets)).sum() / num_fg
    loss_cls = (self.bcewithlog_loss(
        cls_preds.view(-1, self.num_classes)[fg_masks],
        cls_targets)).sum() / num_fg
    if self.use_l1:
      loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks],
                              l1_targets)).sum() / num_fg
    else:
      loss_l1 = 0.0

    reg_weight = 5.0
    loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

    return (
        loss,
        reg_weight * loss_iou,
        loss_obj,
        loss_cls,
        loss_l1,
        num_fg / max(num_gts, 1),
    )

  def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
    l1_target[:, 0] = gt[:, 0] / stride - x_shifts
    l1_target[:, 1] = gt[:, 1] / stride - y_shifts
    l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
    l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
    return l1_target

  @torch.no_grad()
  def get_assignments(
      self,
      batch_idx,
      num_gt,
      total_num_anchors,
      gt_bboxes_per_image,
      gt_classes,
      bboxes_preds_per_image,
      expanded_strides,
      x_shifts,
      y_shifts,
      cls_preds,
      bbox_preds,
      obj_preds,
      labels,
      imgs,
      mode="gpu",
  ):

    if mode == "cpu":
      print("------------CPU Mode for This Batch-------------")
      gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
      bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
      gt_classes = gt_classes.cpu().float()
      expanded_strides = expanded_strides.cpu().float()
      x_shifts = x_shifts.cpu()
      y_shifts = y_shifts.cpu()

    img_size = imgs.shape[2:]
    fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
        gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
        total_num_anchors, num_gt, img_size)

    bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
    cls_preds_ = cls_preds[batch_idx][fg_mask]
    obj_preds_ = obj_preds[batch_idx][fg_mask]
    num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

    if mode == "cpu":
      gt_bboxes_per_image = gt_bboxes_per_image.cpu()
      bboxes_preds_per_image = bboxes_preds_per_image.cpu()

    # pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
    pair_wise_ious = iou(box_convert(gt_bboxes_per_image, 'xywh', 'xyxy'),
                         box_convert(bboxes_preds_per_image))

    gt_cls_per_image = (F.one_hot(gt_classes.to(torch.int64),
                                  self.num_classes).float().unsqueeze(1).repeat(
                                      1, num_in_boxes_anchor, 1))
    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

    if mode == "cpu":
      cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

    with torch.cuda.amp.autocast(enabled=False):
      cls_preds_ = (
          cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() *
          obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())
      pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(),
                                                  gt_cls_per_image,
                                                  reduction="none").sum(-1)
    del cls_preds_

    cost = (pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 *
            (~is_in_boxes_and_center))

    (
        num_fg,
        gt_matched_classes,
        pred_ious_this_matching,
        matched_gt_inds,
    ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt,
                                fg_mask)
    del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

    if mode == "cpu":
      gt_matched_classes = gt_matched_classes.cuda()
      fg_mask = fg_mask.cuda()
      pred_ious_this_matching = pred_ious_this_matching.cuda()
      matched_gt_inds = matched_gt_inds.cuda()

    return (
        gt_matched_classes,
        fg_mask,
        pred_ious_this_matching,
        matched_gt_inds,
        num_fg,
    )

  def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts,
                        y_shifts, total_num_anchors, num_gt, img_size):
    expanded_strides_per_image = expanded_strides[0]
    x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
    y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
    x_centers_per_image = (
        (x_shifts_per_image +
         0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
    )  # [n_anchor] -> [n_gt, n_anchor]
    y_centers_per_image = (
        (y_shifts_per_image +
         0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))

    gt_bboxes_per_image_l = (
        (gt_bboxes_per_image[:, 0] -
         0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(
             1, total_num_anchors))
    gt_bboxes_per_image_r = (
        (gt_bboxes_per_image[:, 0] +
         0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(
             1, total_num_anchors))
    gt_bboxes_per_image_t = (
        (gt_bboxes_per_image[:, 1] -
         0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(
             1, total_num_anchors))
    gt_bboxes_per_image_b = (
        (gt_bboxes_per_image[:, 1] +
         0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(
             1, total_num_anchors))

    b_l = x_centers_per_image - gt_bboxes_per_image_l
    b_r = gt_bboxes_per_image_r - x_centers_per_image
    b_t = y_centers_per_image - gt_bboxes_per_image_t
    b_b = gt_bboxes_per_image_b - y_centers_per_image
    bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
    # in fixed center

    center_radius = 2.5
    # clip center inside image
    gt_bboxes_per_image_clip = gt_bboxes_per_image[:, 0:2].clone()
    gt_bboxes_per_image_clip[:, 0] = torch.clamp(gt_bboxes_per_image_clip[:, 0],
                                                 min=0,
                                                 max=img_size[1])
    gt_bboxes_per_image_clip[:, 1] = torch.clamp(gt_bboxes_per_image_clip[:, 1],
                                                 min=0,
                                                 max=img_size[0])

    gt_bboxes_per_image_l = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(
        1).repeat(1, total_num_anchors
                 ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_r = (gt_bboxes_per_image_clip[:, 0]).unsqueeze(
        1).repeat(1, total_num_anchors
                 ) + center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_t = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(
        1).repeat(1, total_num_anchors
                 ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_b = (gt_bboxes_per_image_clip[:, 1]).unsqueeze(
        1).repeat(1, total_num_anchors
                 ) + center_radius * expanded_strides_per_image.unsqueeze(0)

    c_l = x_centers_per_image - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0

    # in boxes and in centers
    is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

    is_in_boxes_and_center = (is_in_boxes[:, is_in_boxes_anchor] &
                              is_in_centers[:, is_in_boxes_anchor])
    del gt_bboxes_per_image_clip
    return is_in_boxes_anchor, is_in_boxes_and_center

  def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt,
                         fg_mask):
    # Dynamic K
    # ---------------------------------------------------------------
    matching_matrix = torch.zeros_like(cost)

    ious_in_boxes_matrix = pair_wise_ious
    n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
    topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    for gt_idx in range(num_gt):
      _, pos_idx = torch.topk(cost[gt_idx],
                              k=dynamic_ks[gt_idx].item(),
                              largest=False)
      matching_matrix[gt_idx][pos_idx] = 1.0

    del topk_ious, dynamic_ks, pos_idx

    anchor_matching_gt = matching_matrix.sum(0)
    if (anchor_matching_gt > 1).sum() > 0:
      cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
      matching_matrix[:, anchor_matching_gt > 1] *= 0.0
      matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
    fg_mask_inboxes = matching_matrix.sum(0) > 0.0
    num_fg = fg_mask_inboxes.sum().item()

    fg_mask[fg_mask.clone()] = fg_mask_inboxes

    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]

    pred_ious_this_matching = (matching_matrix *
                               pair_wise_ious).sum(0)[fg_mask_inboxes]
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


class YOLOX(nn.Module):
  """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

  def __init__(self, backbone=None, head=None):
    super().__init__()
    if backbone is None:
      backbone = YOLOPAFPN()
    if head is None:
      head = YOLOXHead(80)

    self.backbone = backbone
    self.head = head
    self.init_yolo()

  def init_yolo(self):
    for m in self.modules():
      if isinstance(m, nn.BatchNorm2d):
        m.eps = 1e-3
        m.momentum = 0.03
    self.head.initialize_biases(1e-2)

  def forward(self, x, targets=None):
    # fpn output content features of [dark3, dark4, dark5]
    fpn_outs = self.backbone(x)

    if self.training:
      assert targets is not None
      loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
          fpn_outs, targets, x)
      outputs = {
          "total_loss": loss,
          "iou_loss": iou_loss,
          "l1_loss": l1_loss,
          "conf_loss": conf_loss,
          "cls_loss": cls_loss,
          "num_fg": num_fg,
      }
    else:
      outputs = self.head(fpn_outs)

    return outputs
  
  


@BACKBONE_REGISTRY.register()
def yolox(cfg):
  in_channels = [256, 512, 1024]
  depth = 1.33
  width = 1.25
  conf_thresh = 0.01
  num_classes = cfg.MODEL.NUM_CLASS
  nms_thresh = cfg.TRACKER.NMS_THRESH
  backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
  head = YOLOXHead(num_classes, width, in_channels=in_channels, conf_thresh=conf_thresh,
                   nms_thresh=nms_thresh)
  model = YOLOX(backbone, head)

  return model
