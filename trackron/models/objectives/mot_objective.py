from .base_objective import BaseObjective
import torch
import gc
import random
import torch.nn.functional as F
from torchvision.ops import box_convert
import trackron.models.box_heads.box_regression as boxtrans
from trackron.config import configurable
from trackron.config import configurable
from trackron.models.objectives.losses.hungarian_set import SetCriterion
from trackron.models.objectives.losses.matcher import build_matcher, HungarianMatcher

from .build import OBJECTIVE_REGISTRY
from .losses import BOX_LOSS_REGISTRY, CLS_LOSS_REGISTRY
from .sot_objective import SequenceSOTObjective


@OBJECTIVE_REGISTRY.register()
class MOTObjective(BaseObjective):
  """Objective for training the MOT network."""

  @configurable
  def __init__(self, loss_func, loss_weight=None):
    super(MOTObjective, self).__init__(loss_func, loss_weight)
    if loss_weight is None:
      self.loss_weight = {'target_box_loss': 1.0, 'search_box_loss': 1.0}

  @classmethod
  def from_config(cls, cfg):
    num_classes = cfg.NUM_CLASS
    iteration = cfg.ITERATIONS
    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {
        'detection_loss_ce': cfg.WEIGHT.LOSS_CE,
        'detection_loss_bbox': cfg.WEIGHT.LOSS_BBOX,
        'detection_loss_giou': cfg.WEIGHT.LOSS_GIOU,
        'track_loss_ce': cfg.WEIGHT.LOSS_CE_TRACK,
        'track_loss_bbox': cfg.WEIGHT.LOSS_BBOX_TRACK,
        'track_loss_giou': cfg.WEIGHT.LOSS_GIOU_TRACK
    }
    # TODO this is a hack
    aux_weight_dict = {}
    for i in range(iteration - 1):
      aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    # aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    loss_func = SetCriterion(num_classes, matcher, losses)
    return {"loss_func": loss_func, "loss_weight": weight_dict}

  def forward(self, results, targets):
    """
        args:
            data - The input data, should contain the fields 'template_images', 'search_images', 'train_anno',
                    'search_proposals', 'proposal_iou' and 'search_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
    det_preds = {
        'pred_logits': results['pred_logits'][-1],
        'pred_boxes': results['pred_boxes'][-1]
    }
    det_preds['aux_outputs'] = [{k: v[i]
                                 for k, v in results.items()}
                                for i in range(len(results['pred_logits']) - 1)]
    det_losses = self.loss_func(det_preds, targets['det_targets'])

    track_preds = {
        'pred_logits': results['track_logits'][-1],
        'pred_boxes': results['track_boxes'][-1]
    }
    track_preds['aux_outputs'] = [{
        'pred_logits': results['track_logits'][i],
        'pred_boxes': results['track_boxes'][i]
    } for i in range(len(results['track_logits']) - 1)]
    track_losses = self.loss_func(track_preds, targets['track_targets'])

    losses, weighted_losses, metrics = {}, {}, {}

    for k in det_losses:
      if k.startswith('loss'):
        losses[f'DetLoss/{k}'] = det_losses[k]
        losses[f'TrackLoss/{k}'] = track_losses[k]
        weighted_losses[
            f'DetLoss/{k}'] = det_losses[k] * self.loss_weight[f'detection_{k}']
        weighted_losses[
            f'TrackLoss/{k}'] = track_losses[k] * self.loss_weight[f'track_{k}']
      else:
        metrics[f'metrics/det_{k}'] = det_losses[k]
        metrics[f'metrics/track_{k}'] = track_losses[k]

    return losses, weighted_losses, metrics


@OBJECTIVE_REGISTRY.register()
class MOTObjective2(BaseObjective):
  """Objective for training the MOT network. Support box center score"""

  @configurable
  def __init__(self, loss_func, loss_weight=None):
    super(MOTObjective2, self).__init__(loss_func, loss_weight)
    if loss_weight is None:
      self.loss_weight = {'target_box_loss': 1.0, 'search_box_loss': 1.0}

  @classmethod
  def from_config(cls, cfg):
    num_classes = cfg.NUM_CLASS
    iteration = cfg.ITERATIONS
    weight_dict = {
        'detection_loss_ce': cfg.WEIGHT.LOSS_CE,
        'detection_loss_bbox': cfg.WEIGHT.LOSS_BBOX,
        'detection_loss_giou': cfg.WEIGHT.LOSS_GIOU,
        'track_loss_ce': cfg.WEIGHT.LOSS_CE_TRACK,
        'track_loss_bbox': cfg.WEIGHT.LOSS_BBOX_TRACK,
        'track_loss_giou': cfg.WEIGHT.LOSS_GIOU_TRACK,
        'track_loss_center': cfg.WEIGHT.LOSS_CENTER_TRACK
    }
    # TODO this is a hack
    aux_weight_dict = {}
    for i in range(iteration - 1):
      aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    # aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    det_losses = ['labels', 'boxes', 'cardinality']
    det_matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    det_loss_func = SetCriterion(num_classes, det_matcher, det_losses)
    track_losses = ['centers', 'boxes', 'cardinality']
    track_matcher = HungarianMatcher(cost_class=0.0,
                                     cost_bbox=5.0,
                                     cost_giou=2.0)
    track_loss_func = SetCriterion(num_classes, track_matcher, track_losses)
    return {
        "loss_func": {
            "detection": det_loss_func,
            "track": track_loss_func
        },
        "loss_weight": weight_dict
    }

  def forward(self, results, targets):
    """
        args:
            data - The input data, should contain the fields 'template_images', 'search_images', 'train_anno',
                    'search_proposals', 'proposal_iou' and 'search_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
    losses, weighted_losses, metrics = {}, {}, {}
    det_preds = {
        'pred_logits': results['pred_logits'][-1],
        'pred_boxes': results['pred_boxes'][-1]
    }
    det_preds['aux_outputs'] = [{k: v[i]
                                 for k, v in results.items()}
                                for i in range(len(results['pred_logits']) - 1)]
    det_losses = self.loss_func['detection'](det_preds, targets['det_targets'])
    for k in det_losses:
      if k.startswith('loss'):
        losses[f'DetLoss/{k}'] = det_losses[k]
        weighted_losses[
            f'DetLoss/{k}'] = det_losses[k] * self.loss_weight[f'detection_{k}']
      else:
        metrics[f'metrics/det_{k}'] = det_losses[k]

    track_preds = {
        'pred_logits': results['track_logits'][-1],
        'pred_boxes': results['track_boxes'][-1],
        'pred_centers': results['track_centers'][-1]
    }
    track_preds['aux_outputs'] = [{
        'pred_logits': results['track_logits'][i],
        'pred_boxes': results['track_boxes'][i],
        'pred_centers': results['track_centers'][i]
    } for i in range(len(results['track_logits']) - 1)]
    track_losses = self.loss_func['track'](track_preds,
                                           targets['track_targets'])
    for k in track_losses:
      if k.startswith('loss'):
        losses[f'TrackLoss/{k}'] = track_losses[k]
        weighted_losses[
            f'TrackLoss/{k}'] = track_losses[k] * self.loss_weight[f'track_{k}']
      else:
        metrics[f'metrics/track_{k}'] = track_losses[k]

    return losses, weighted_losses, metrics


@OBJECTIVE_REGISTRY.register()
class MOTObjective3(BaseObjective):
  """Objective for training the MOT network."""

  @configurable
  def __init__(self, loss_func, loss_weight=None):
    super(MOTObjective3, self).__init__(loss_func, loss_weight)
    if loss_weight is None:
      self.loss_weight = {'target_box_loss': 1.0, 'search_box_loss': 1.0}

  @classmethod
  def from_config(cls, cfg):
    num_classes = cfg.NUM_CLASS
    iteration = cfg.ITERATIONS
    weight_dict = {
        'detection_loss_ce': cfg.WEIGHT.LOSS_CE,
        'detection_loss_bbox': cfg.WEIGHT.LOSS_BBOX,
        'detection_loss_giou': cfg.WEIGHT.LOSS_GIOU,
        'detection_loss_mask': cfg.WEIGHT.LOSS_MASK,
        'track_loss_ce': cfg.WEIGHT.LOSS_CE_TRACK,
        'track_loss_bbox': cfg.WEIGHT.LOSS_BBOX_TRACK,
        'track_loss_giou': cfg.WEIGHT.LOSS_GIOU_TRACK,
        'track_loss_center': cfg.WEIGHT.LOSS_CENTER_TRACK
    }
    # TODO this is a hack
    aux_weight_dict = {}
    for i in range(iteration - 1):
      aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    # aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    if cfg.WEIGHT.LOSS_MASK == 0:
      det_losses = ['labels', 'boxes', 'cardinality']
    else:
      det_losses = ['labels', 'boxes', 'masks', 'cardinality']
    det_matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    det_loss_func = SetCriterion(num_classes, det_matcher, det_losses)
    track_loss_func = SequenceSOTObjective(cfg)
    return {
        "loss_func": {
            "detection": det_loss_func,
            "track": track_loss_func
        },
        "loss_weight": weight_dict
    }

  def forward(self, results, targets):
    """
        args:
            data - The input data, should contain the fields 'template_images', 'search_images', 'train_anno',
                    'search_proposals', 'proposal_iou' and 'search_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """

    losses, weighted_losses, metrics = {}, {}, {}
    ### detection loss
    if results.get('pred_boxes', None) is not None:
      det_preds = {
          'pred_logits': results['pred_logits'][-1],
          'pred_boxes': results['pred_boxes'][-1]
      }
      if 'pred_masks' in results:
        det_preds['pred_masks'] = results['pred_masks'][-1]
      det_preds['aux_outputs'] = [{k: results[k][i]
                                  for k in det_preds}
                                  for i in range(len(results['pred_logits']) - 1)]
      det_losses = self.loss_func['detection'](det_preds, targets['det_targets'])
      for k in det_losses:
        if k.startswith('loss'):
          losses[f'DetLoss/{k}'] = det_losses[k]
          weighted_losses[
              f'DetLoss/{k}'] = det_losses[k] * self.loss_weight[f'detection_{k}']
        else:
          metrics[f'metrics/det_{k}'] = det_losses[k]

    #### track loss
    valid_ids = targets["valid_preds"]
    track_preds = {
        'pred_boxes': [
            torch.cat(
                [box_convert(boxes[inds], 'cxcywh', 'xyxy')
                 for inds, boxes in zip(valid_ids, batch_boxes)])
            for batch_boxes in results['track_boxes']
        ],
    }
    if len(results.get('track_scores', [])) > 1:
      track_preds['pred_scores'] = [
            torch.cat(
                [scores[inds] for inds, scores in zip(valid_ids, batch_scores)])
            for batch_scores in results['track_scores']
        ]
    track_targets = {'search_boxes': box_convert(torch.cat(targets['valid_track_boxes']), 'cxcywh', 'xyxy')}
    track_losses, weighted_track_losses, track_metrics = self.loss_func[
        'track'](track_preds, track_targets, normalize_box=False)
    for k in track_losses:
      if k.startswith('loss'):
        loss_name = k.split('/')[-1]
        losses[f'TrackLoss/{loss_name}'] = track_losses[k]
        weighted_losses[f'TrackLoss/{loss_name}'] = weighted_track_losses[k]
      else:
        loss_name = k.split('/')[-1]
        metrics[f'metrics/track_{loss_name}'] = track_losses[k]

    return losses, weighted_losses, metrics
