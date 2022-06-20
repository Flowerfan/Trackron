from .base_objective import BaseObjective
import torch
import gc
import random
import torch.nn.functional as F
from torchvision.ops import box_convert
from functools import partial

import trackron.models.box_heads.box_regression as boxtrans
from trackron.config import configurable
from trackron.config import configurable
from trackron.models.objectives.losses import giou_loss, LovaszSegLoss
from trackron.evaluation.analysis.vos_utils import davis_jaccard_measure
from trackron.data.utils import normalize_boxes
from trackron.data.mask_ops import crop_and_resize

from .build import OBJECTIVE_REGISTRY
from .losses import BOX_LOSS_REGISTRY, CLS_LOSS_REGISTRY, centerness_loss


@OBJECTIVE_REGISTRY.register()
class SOTObjective(BaseObjective):
  """Objective for training the DiMP network."""

  @configurable
  def __init__(self, loss_func, loss_weight=None):
    super(SOTObjective, self).__init__(loss_func, loss_weight)
    if loss_weight is None:
      self.loss_weight = {'loss_bbox': 1.0, 'loss_giou': 1.0}

  @classmethod
  def from_config(cls, cfg):
    # loss_func = {"loss_bbox": F.l1_loss, "loss_giou": giou_loss, "loss_mask": LovaszSegLoss()}
    loss_func = {
        "loss_bbox":
            F.l1_loss,
        "loss_giou":
            giou_loss,
        "loss_mask":
            partial(F.binary_cross_entropy_with_logits, reduction='mean')
    }
    loss_weight = {
        "loss_bbox": cfg.WEIGHT.LOSS_BBOX,
        "loss_giou": cfg.WEIGHT.LOSS_GIOU,
        "loss_mask": cfg.WEIGHT.LOSS_MASK,
    }
    return {"loss_func": loss_func, "loss_weight": loss_weight}

  def calculate_box_losses(self, pred_boxes, gt_boxes):
    # l1_loss = self.loss_func['loss_bbox'](pred_boxes, gt_boxes)
    l1_loss = self.loss_func['loss_bbox'](box_convert(pred_boxes, 'xyxy',
                                                      'cxcywh'),
                                          box_convert(gt_boxes, 'xyxy',
                                                      'cxcywh'),
                                          reduction='none')
    l1_loss = l1_loss.sum(1).mean()
    try:
      giou_loss, iou = self.loss_func['loss_giou'](pred_boxes, gt_boxes)
    except:
      giou_loss, iou = torch.tensor(1.0).to(
          pred_boxes.device), torch.tensor(0.0).to(pred_boxes.device)
    return l1_loss, giou_loss, iou

  def calculate_mask_losses(self, pred_masks, gt_masks):
    mask_loss = self.loss_func['loss_mask'](pred_masks, gt_masks)
    acc = cnt = 0
    acc_l = [
        davis_jaccard_measure(
            torch.sigmoid(rm.detach()).cpu().numpy() > 0.5,
            lb.cpu().numpy())
        for rm, lb in zip(pred_masks.view(-1, *pred_masks.shape[-2:]),
                          gt_masks.view(-1, *pred_masks.shape[-2:]))
    ]
    acc += sum(acc_l)
    cnt += len(acc_l)
    acc = torch.tensor(acc / cnt,
                       dtype=mask_loss.dtype,
                       device=mask_loss.device)
    return mask_loss, acc

  def forward(self, results, data, normalize_box=True):
    """
        args:
            data - The Groundturth data, should contain the fields 'search_boxes', 'search_images'

        returns:
            losses   - the training loss dict
            weighted_losses  -  weighted loss dict
            metrics - evaluation training process
    """
    pred_boxes = results['pred_boxes'].reshape(-1, 4)
    gt_boxes = data['search_boxes'].reshape(-1, 4).to(pred_boxes.device)
    if normalize_box:
      norm_box = torch.tensor(data['search_images'].shape[-2:],
                              dtype=torch.float32,
                              device=pred_boxes.device).flip(-1).repeat(2)[None]
      gt_boxes = gt_boxes / norm_box
    # target_box = box_xywh_to_xyxy(target_box)

    # box prediction use on-frame feature
    l1_loss, giou_loss, iou = self.calculate_box_losses(pred_boxes, gt_boxes)

    # Loss of the final filter
    losses = {
        'losses/box_l1_loss': l1_loss,
        'losses/box_giou_loss': giou_loss,
    }
    weighted_losses = {
        'losses/box_l1_loss': l1_loss * self.loss_weight['loss_bbox'],
        'losses/box_giou_loss': giou_loss * self.loss_weight['loss_giou'],
    }
    metrics = {'metrics/IoU': iou}

    return losses, weighted_losses, metrics


@OBJECTIVE_REGISTRY.register()
class SOTObjectiveMultiQuery(SOTObjective):
  """Objective for training the Squence predict boxes"""

  def forward(self, results, data):
    assert results['pred_boxes'].ndim == 3
    pred_boxes = results['pred_boxes']
    H, W = data['search_images'].shape[-2:]
    norm_box = torch.tensor([[W, H, W, H]],
                            dtype=torch.float32,
                            device=pred_boxes.device)
    gt_boxes = data['search_boxes'].reshape(-1, 4).to(
        pred_boxes.device) / norm_box
    # target_box = box_xywh_to_xyxy(target_box)

    # box prediction use on-frame feature
    l1_loss = sum([
        self.loss_func['loss_bbox'](boxes, gt_boxes) for boxes in pred_boxes
    ]) / len(pred_boxes)
    try:
      giou_loss, iou = sum([
          self.loss_func['loss_giou'](boxes, gt_boxes) for boxes in pred_boxes
      ]) / len(pred_boxes)
    except:
      giou_loss, iou = torch.tensor(1.0).to(
          pred_boxes.device), torch.tensor(0.0).to(pred_boxes.device)

    # Loss of the final filter
    losses = {
        'losses/box_l1_loss': l1_loss,
        'losses/box_giou_loss': giou_loss,
        # 'losses/search_box_l1_loss': search_box_loss_l1,
        # 'losses/search_box_giou_loss': search_box_loss_giou,
    }
    weighted_losses = {
        'losses/box_l1_loss': l1_loss * self.loss_weight['loss_bbox'],
        'losses/box_giou_loss': giou_loss * self.loss_weight['loss_giou'],
    }
    metrics = {'metrics/IoU': iou}

    return losses, weighted_losses, metrics


@OBJECTIVE_REGISTRY.register()
class SequenceSOTObjective(SOTObjective):
  """Objective for training the Squence predict boxes"""

  def forward(self, results, data, normalize_box=True):
    pred_boxes = results['pred_boxes']
    gt_boxes = data['search_boxes'].reshape(-1, 4).to(pred_boxes[0].device)
    if normalize_box:
      norm_box = torch.tensor(
          data['search_images'].shape[-2:],
          dtype=torch.float32,
          device=pred_boxes[0].device).flip(-1).repeat(2)[None]
      gt_boxes = gt_boxes / norm_box

    # box prediction use on-frame feature
    l1_losses = giou_losses = ious = 0
    for boxes in pred_boxes:
      l1_loss, giou_loss, iou = self.calculate_box_losses(
          boxes.view(-1, 4), gt_boxes)
      l1_losses += l1_loss
      giou_losses += giou_loss
      ious += iou
    l1_loss = l1_losses / len(pred_boxes)
    giou_loss = giou_losses / len(pred_boxes)
    iou = ious / len(pred_boxes)

    # Loss of the boxes
    losses = {
        'losses/box_l1_loss': l1_loss,
        'losses/box_giou_loss': giou_loss,
    }
    weighted_losses = {
        'losses/box_l1_loss': l1_loss * self.loss_weight['loss_bbox'],
        'losses/box_giou_loss': giou_loss * self.loss_weight['loss_giou'],
    }
    ### box score loss
    pred_scores = results.get('pred_scores', None)
    if pred_scores is not None and len(pred_scores) > 0 and giou_loss.item() < 1.0:
      center_loss = torch.zeros_like(iou)
      for boxes, scores in zip(pred_boxes[-len(pred_scores):], pred_scores):
        center_loss += centerness_loss(scores.flatten(), boxes.view(-1, 4),
                                       gt_boxes).mean()
      center_loss /= len(pred_scores)
      losses['losses/box_score_loss'] = center_loss
      weighted_losses['losses/box_score_loss'] = center_loss * 1.0

    metrics = {'metrics/IoU': iou}

    ### mask loss
    pred_masks = results.get('pred_masks', None)
    if pred_masks is not None and giou_loss.item() < 1.0:
      mask_loss, mask_metric = self.mask_loss(
          pred_masks, pred_boxes, data['search_target_masks'].flatten(0, 1))
      losses['losses/mask_loss'] = mask_loss
      weighted_losses[
          'losses/mask_loss'] = mask_loss * self.loss_weight['loss_mask']
      metrics['metrics/Mask_Acc'] = mask_metric

    return losses, weighted_losses, metrics

  def mask_loss(self, pred_masks, pred_boxes, gt_masks):
    mask_loss = mask_metric = 0
    num_pred = len(pred_masks)
    image_sz = gt_masks.shape[-2:]
    for pred_box, pred_mask in zip(pred_boxes[-num_pred:], pred_masks):
      mask_size = pred_mask.size(-1)
      pred_box = normalize_boxes(pred_box.view(-1, 4),
                                 image_sz,
                                 in_format='xyxy',
                                 out_format='xyxy',
                                 reverse=True)
      gt_mask = crop_and_resize(gt_masks, pred_box, mask_size).float()
      loss, metric = self.calculate_mask_losses(
          pred_mask.view(-1, mask_size, mask_size), gt_mask)
      mask_loss += loss
      mask_metric += metric
    return mask_loss / len(pred_masks), mask_metric / len(pred_masks)