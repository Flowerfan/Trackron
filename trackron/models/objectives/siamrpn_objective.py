import torch
import torch.nn.functional as F
from trackron.config import configurable


from .base_objective import BaseObjective
from .build import OBJECTIVE_REGISTRY


def get_cls_loss(pred, label, select):
  if len(select.size()) == 0 or \
          select.size() == torch.Size([0]):
    return 0
  pred = torch.index_select(pred, 0, select)
  label = torch.index_select(label, 0, select)
  return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
  pred = pred.view(-1, 2)
  label = label.view(-1)
  pos = label.data.eq(1).nonzero().squeeze().cuda()
  neg = label.data.eq(0).nonzero().squeeze().cuda()
  loss_pos = get_cls_loss(pred, label, pos)
  loss_neg = get_cls_loss(pred, label, neg)
  return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
  b, _, sh, sw = pred_loc.size()
  pred_loc = pred_loc.view(b, 4, -1, sh, sw)
  diff = (pred_loc - label_loc).abs()
  diff = diff.sum(dim=1).view(b, -1, sh, sw)
  loss = diff * loss_weight
  return loss.sum().div(b)


@OBJECTIVE_REGISTRY.register()
class SiamRPNObjective(BaseObjective):
  """Objective for training the SiamRPN network."""

  @configurable
  def __init__(self, loss_func, loss_weight=None):
    super(SiamRPNObjective, self).__init__(loss_func, loss_weight)
    assert "cls_loss" in loss_func
    assert "box_loss" in loss_func
    if loss_weight is None:
      self.loss_weight = {'cls_loss': 1.0, 'box_loss': 1.0}

  @classmethod
  def from_config(cls, cfg):
    loss_weight = {
        "cls_loss": cfg.WEIGHT.LOSS_CLS,
        "box_loss": cfg.WEIGHT.LOSS_BBOX
    }
    loss_func = {'cls_loss': select_cross_entropy_loss,
                 'box_loss': weight_l1_loss}
    return {"loss_func": loss_func, "loss_weight": loss_weight}

  def forward(self, results, data):
    """
        args:
            data - The input data, should contain the fields 'template_images', 'search_images', 'train_anno',
                    'search_proposals', 'proposal_iou' and 'search_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
    # Run network
    pred_scores = results['pred_scores']
    pred_boxes = results['pred_boxes']

    gt_scores = data['search_label'].to(pred_scores.device)
    gt_boxes = data['search_boxes_delta'].to(pred_boxes.device)
    gt_boxes_weight = data['search_boxes_delta_weight'].to(pred_boxes.device)

    # Classification losses for the different optimization iterations
    cls_loss = self.loss_func['cls_loss'](pred_scores, gt_scores)
    box_loss = self.loss_func['box_loss'](pred_boxes, gt_boxes, gt_boxes_weight)
    
    # Loss of the final filter
    losses = {
        'losses/loss_box': box_loss,
        'losses/loss_cls': cls_loss
    }
    weighted_losses = {
        'losses/loss_box_iou': box_loss * self.loss_weight['box_loss'],
        'losses/loss_cls': cls_loss * self.loss_weight['cls_loss']
    }

    return losses, weighted_losses, None
