import torch.nn as nn
import torch
from torch.nn import functional as F
from trackron.config import configurable
from .build import CLS_LOSS_REGISTRY

@CLS_LOSS_REGISTRY.register()
class LBHinge(nn.Module):

  @configurable
  def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
    """Loss that uses a 'hinge' on the lower bound.
        This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
        also smaller than that threshold.
        args:
            error_matric:  What base loss to use (MSE by default).
            threshold:  Threshold to use for the hinge.
            clip:  Clip the loss if it is above this value.
        """
    super().__init__()
    self.error_metric = error_metric
    self.threshold = threshold if threshold is not None else -100
    self.clip = clip

  @classmethod
  def from_config(cls, cfg):
    return {"threshold": cfg.LOSS_CLS_THRESHOLD}

  def forward(self, prediction, label):
    negative_mask = (label < self.threshold).float()
    positive_mask = (1.0 - negative_mask)

    prediction = negative_mask * \
        F.relu(prediction) + positive_mask * prediction

    loss = self.error_metric(prediction, positive_mask * label)

    if self.clip is not None:
      loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
    return loss


@CLS_LOSS_REGISTRY.register()
class ContraLoss(nn.Module):
  """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

  @configurable
  def __init__(self, threshold=0.05, clip=None):
    super().__init__()
    # self.error_metric = error_metric
    self.threshold = threshold
    self.clip = clip

  @classmethod
  def from_config(cls, cfg):
    return {"threshold": cfg.LOSS_CLS_THRESHOLD}

  def forward(self, prediction, label):
    negative_mask = (label < self.threshold).float()
    positive_mask = (1.0 - negative_mask)

    # shift = prediction.mean(dim=(-2,-1), keepdim=True)
    shift = prediction.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    p_exp = (prediction - shift).exp()
    # log_sum = torch.logsumexp(prediction.view, dim=(-2,-1))
    log_sum = torch.log(p_exp.sum(dim=(-2, -1)).clamp_(1e-7))
    log_pos = torch.log((p_exp * positive_mask).sum(dim=(-2, -1)).clamp_(1e-7))
    loss = (log_sum - log_pos).mean()

    if self.clip is not None:
      loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
    return loss




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
    pos = label.data.eq(1).nonzero(as_tuple=False).squeeze().cuda()
    neg = label.data.eq(0).nonzero(as_tuple=False).squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5
