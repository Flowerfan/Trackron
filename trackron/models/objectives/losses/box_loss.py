import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from trackron.config import configurable
from .build import BOX_LOSS_REGISTRY
from torchvision.ops.boxes import box_area


@BOX_LOSS_REGISTRY.register()
class IoUScore(nn.MSELoss):

  @configurable
  def __init__(self):
    """[summary]
        """
    super(IoUScore, self).__init__()

  @classmethod
  def from_config(cls, cfg):
    return {}

  def forward(self, iou_pred, data):
    return super(IoUScore, self).forward(iou_pred, data['proposal_iou'])


@BOX_LOSS_REGISTRY.register()
class KLRegression(nn.Module):
  """KL-divergence loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

  @configurable
  def __init__(self, eps=0.0):
    """[summary]
    Args:
        eps (float, optional): [description]. Defaults to 0.0.
    """
    super().__init__()
    self.eps = eps

  @classmethod
  def from_config(cls, cfg):
    return {}

  def forward(self, scores, data):
    """Args:
            scores: predicted score values
            sample_density: probability density of the sample distribution
            gt_density: probability density of the ground truth distribution
            mc_dim: dimension of the MC samples"""
    mc_dim = 1
    is_valid = data['search_boxes'][:, :, 0] < 99999.0
    scores = scores[is_valid, :]
    sample_density = data['proposal_density'][is_valid, :]
    gt_density = data['gt_density'][is_valid, :]

    exp_val = scores - torch.log(sample_density + self.eps)

    L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim]) - \
        torch.mean(scores * (gt_density / (sample_density + self.eps)), dim=mc_dim)

    return L.mean()


class IoULoss(nn.Module):

  def __init__(self):
    super(IoULoss, self).__init__()

  def cal_iou_loss(self, pred, target, weight=None):
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * \
        (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
        torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
        torch.min(pred_top, target_top)

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

    if weight is not None and weight.sum() > 0:
      return (losses * weight).sum() / weight.sum()
    else:
      assert losses.numel() != 0
      return losses.mean()

  def forward(self, bbox_pred, reg_target, reg_weight):
    """
            :param bbox_pred:
            :param reg_target:
            :param reg_weight:
            :param grid_x:  used to get real target bbox
            :param grid_y:  used to get real target bbox
            :return:
            """

    bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)
    reg_target_flatten = reg_target.view(-1, 4)
    reg_weight_flatten = reg_weight.view(-1)
    pos_inds = torch.nonzero(reg_weight_flatten > 0, as_tuple=False).squeeze(1)

    bbox_pred_flatten = bbox_pred_flatten[pos_inds]
    reg_target_flatten = reg_target_flatten[pos_inds]

    loss = self.cal_iou_loss(bbox_pred_flatten, reg_target_flatten)

    return loss

  def pred_to_image(self, bbox_pred):
    grid_to_search_x = self.grid_to_search_x.to(bbox_pred.device)
    grid_to_search_y = self.grid_to_search_y.to(bbox_pred.device)

    pred_x1 = grid_to_search_x - \
        bbox_pred[:, 0, ...].unsqueeze(1)
    pred_y1 = grid_to_search_y - \
        bbox_pred[:, 1, ...].unsqueeze(1)
    pred_x2 = grid_to_search_x + \
        bbox_pred[:, 2, ...].unsqueeze(1)
    pred_y2 = grid_to_search_y + \
        bbox_pred[:, 3, ...].unsqueeze(1)

    pred = [pred_x1, pred_y1, pred_x2, pred_y2]

    pred = torch.cat(pred, dim=1)

    return pred


def smooth_l1_loss(input: torch.Tensor,
                   target: torch.Tensor,
                   beta: float,
                   reduction: str = "none") -> torch.Tensor:
  """
    Smooth L1 loss defined in the Fast R-CNN paper as:

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.

    Smooth L1 loss is related to Huber loss, which is defined as:

                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise

    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:

     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.

    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.

    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
    """
  if beta < 1e-5:
    # if beta == 0, then torch.where will result in nan gradients when
    # the chain rule is applied due to pytorch implementation details
    # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
    # zeros, rather than "no gradient"). To avoid this issue, we define
    # small values of beta to be exactly l1 loss.
    loss = torch.abs(input - target)
  else:
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

  if reduction == "mean":
    loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
  elif reduction == "sum":
    loss = loss.sum()
  return loss


# def giou_loss(
#     boxes1: torch.Tensor,
#     boxes2: torch.Tensor,
#     reduction: str = "none",
#     eps: float = 1e-7,
# ) -> torch.Tensor:
#   """
#     Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
#     https://arxiv.org/abs/1902.09630

#     Gradient-friendly IoU loss with an additional penalty that is non-zero when the
#     boxes do not overlap and scales with the size of their smallest enclosing box.
#     This loss is symmetric, so the boxes1 and boxes2 arguments are interchangeable.

#     Args:
#         boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
#         reduction: 'none' | 'mean' | 'sum'
#                  'none': No reduction will be applied to the output.
#                  'mean': The output will be averaged.
#                  'sum': The output will be summed.
#         eps (float): small number to prevent division by zero
#     """

#   x1, y1, w, h = boxes1.unbind(dim=-1)
#   x1g, y1g, gw, gh = boxes2.unbind(dim=-1)
#   x2 = x1 + w
#   y2 = y1 + h
#   x2g, y2g = x1g + gw, y1g + gh

#   assert (x2 >= x1).all(), "bad box: x1 larger than x2"
#   assert (y2 >= y1).all(), "bad box: y1 larger than y2"

#   # Intersection keypoints
#   xkis1 = torch.max(x1, x1g)
#   ykis1 = torch.max(y1, y1g)
#   xkis2 = torch.min(x2, x2g)
#   ykis2 = torch.min(y2, y2g)

#   intsctk = torch.zeros_like(x1)
#   mask = (ykis2 > ykis1) & (xkis2 > xkis1)
#   intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
#   unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
#   iouk = intsctk / (unionk + eps)

#   # smallest enclosing box
#   xc1 = torch.min(x1, x1g)
#   yc1 = torch.min(y1, y1g)
#   xc2 = torch.max(x2, x2g)
#   yc2 = torch.max(y2, y2g)

#   area_c = (xc2 - xc1) * (yc2 - yc1)
#   miouk = iouk - ((area_c - unionk) / (area_c + eps))

#   loss = 1 - miouk

#   if reduction == "mean":
#     loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
#   elif reduction == "sum":
#     loss = loss.sum()

#   return loss


def box_iou(boxes1, boxes2):
  """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
  area1 = box_area(boxes1)  # (N,)
  area2 = box_area(boxes2)  # (N,)

  lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
  rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

  wh = (rb - lt).clamp(min=0)  # (N,2)
  inter = wh[:, 0] * wh[:, 1]  # (N,)

  union = area1 + area2 - inter

  iou = inter / union
  return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
  """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    boxes1: (N, 4)
    boxes2: (N, 4)
    """
  # degenerate boxes gives inf / nan results
  # so do an early check
  # try:
  assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
  assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
  iou, union = box_iou(boxes1, boxes2)  # (N,)

  lt = torch.min(boxes1[:, :2], boxes2[:, :2])
  rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

  wh = (rb - lt).clamp(min=0)  # (N,2)
  area = wh[:, 0] * wh[:, 1]  # (N,)

  return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
  """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
  giou, iou = generalized_box_iou(boxes1, boxes2)
  return (1 - giou).mean(), iou.mean()


def centerness_loss(center_scores, pred_boxes, gt_boxes):
  """calculate box centerness(confidence) losses

  Args:
      center_scores ([N]): center scores
      pred_boxes ([N 4]): [pred boxes by the model in xyxy format]
      gt_boxes ([type]): [gt boxes in xyxy format]
  """
  cxcy = (pred_boxes[:, :2] + pred_boxes[:, 2:]).detach() / 2
  lt = cxcy - gt_boxes[:, :2]
  rb = gt_boxes[:, 2:] - cxcy
  lr = (torch.minimum(lt[:, 0], rb[:, 0]) /
        torch.maximum(lt[:, 0], rb[:, 0]).clamp(min=1e-6)).clamp(min=0)
  bt = (torch.minimum(lt[:, 1], rb[:, 1]) /
        torch.maximum(lt[:, 1], rb[:, 1]).clamp(min=1e-6)).clamp(min=0)
  gt_scores = (lr * bt).sqrt()
  valid = torch.nonzero(gt_scores)
  centerness = F.binary_cross_entropy_with_logits(center_scores, gt_scores, reduction='none')
  return centerness
