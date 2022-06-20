from .base_objective import BaseObjective
import torch
import gc
import random
import trackron.models.box_heads.box_regression as boxtrans
from trackron.config import configurable
from .build import OBJECTIVE_REGISTRY
from .losses import BOX_LOSS_REGISTRY, CLS_LOSS_REGISTRY
from trackron.config import configurable


@OBJECTIVE_REGISTRY.register()
class DiMPObjective(BaseObjective):
  """Objective for training the DiMP network."""

  @configurable
  def __init__(self, loss_func, loss_weight=None):
    super(DiMPObjective, self).__init__(loss_func, loss_weight)
    assert "cls_loss" in loss_func
    assert "box_loss" in loss_func
    if loss_weight is None:
      self.loss_weight = {'cls_loss': 1.0, 'box_loss': 1.0}

  @classmethod
  def from_config(cls, cfg):
    loss_func = {
        "cls_loss": CLS_LOSS_REGISTRY.get(cfg.LOSS_CLS)(cfg),
        "box_loss": BOX_LOSS_REGISTRY.get(cfg.LOSS_BBOX)(cfg)
    }
    loss_weight = {
        "cls_loss": cfg.WEIGHT.LOSS_CLS,
        "box_loss": cfg.WEIGHT.LOSS_BBOX
    }
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
    target_scores = results['target_scores']
    box_pred = results['box_pred']

    # Classification losses for the different optimization iterations
    cls_losses_search = [
        self.loss_func['cls_loss'](s, data) for s in target_scores
    ]

    # Compute loss for ATOM IoUNet
    box_loss = self.loss_weight['box_loss'] * self.loss_func['box_loss'](
        box_pred, data)

    if isinstance(self.loss_weight['cls_loss'], float) == 1:
      cls_ws = [
          self.loss_weight['cls_loss'] for _ in range(len(cls_losses_search))
      ]
    else:
      assert len(self.loss_weight['cls_loss']) == len(cls_losses_search)
      cls_ws = self.loss_weight['cls_loss']

    # Loss of the final filter
    clf_loss_search = cls_losses_search[-1] * cls_ws[-1]
    losses = {
        'losses/loss_box_iou': box_loss,
        'losses/loss_target_clf': clf_loss_search
    }

    if len(cls_losses_search) > 1:

      # Loss for the initial filter iteration
      losses['losses/loss_init_clf'] = cls_losses_search[0] * cls_ws[0]
      # Loss for the intermediate filter iterations
      losses['losses/loss_iter_clf'] = sum(
          [w * l for w, l in zip(cls_ws[1:-1], cls_losses_search[1:-1])])
    return losses
