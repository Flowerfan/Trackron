import torch
import copy
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from trackron.utils.comm import get_world_size
from torchvision.ops import box_convert, generalized_box_iou
from trackron.structures.masks import BitMasks
from trackron.structures.tensor import nested_tensor_from_tensor_list

from .box_loss import centerness_loss


def sigmoid_focal_loss(inputs,
                       targets,
                       num_boxes,
                       alpha: float = 0.25,
                       gamma: float = 2):
  """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
  prob = inputs.sigmoid()
  ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                               targets,
                                               reduction="none")
  p_t = prob * targets + (1 - prob) * (1 - targets)
  loss = ce_loss * ((1 - p_t)**gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

  return loss.mean(1).sum() / num_boxes


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  if target.numel() == 0:
    return [torch.zeros([], device=output.device)]
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res


class ProcessorDCT(object):

  def __init__(self, n_keep, gt_mask_len):
    self.n_keep = n_keep
    self.gt_mask_len = gt_mask_len

    inputs = np.zeros((self.gt_mask_len, self.gt_mask_len))
    _, zigzag_table = self.zigzag(inputs)
    self.zigzag_table = zigzag_table[:self.n_keep]

  def sigmoid(self, x):
    """Apply the sigmoid operation.
        """
    y = 1. / (1. + 1. / np.exp(x))
    dy = y * (1 - y)
    return y

  def inverse_sigmoid(self, x):
    """Apply the inverse sigmoid operation.
                y = -ln(1-x/x)
        """
    y = -1 * np.log((1 - x) / x)
    return y

  def zigzag(self, input, gt=None):
    """
        Zigzag scan of a matrix
        Argument is a two-dimensional matrix of any size,
        not strictly a square one.
        Function returns a 1-by-(m*n) array,
        where m and n are sizes of an input matrix,
        consisting of its items scanned by a zigzag method.
        
        Args:
            input (np.array): shape [h,w], value belong to [-127, 128], transformed from gt.
            gt (np.array): shape [h,w], value belong to {0,1}, original instance segmentation gt mask.
        Returns:
            output (np.array): zig-zag encoded values, shape [h*w].
            indicator (np.array): positive sample indicator, shape [h,w].
        """
    # initializing the variables
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    vmax = input.shape[0]
    hmax = input.shape[1]
    assert vmax == hmax

    i = 0
    output = np.zeros((vmax * hmax))
    indicator = []

    while ((v < vmax) and (h < hmax)):
      if ((h + v) % 2) == 0:  # going up
        if (v == vmin):
          output[i] = input[v, h]  # if we got to the first line
          indicator.append(v * vmax + h)
          if (h == hmax):
            v = v + 1
          else:
            h = h + 1
          i = i + 1
        elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
          output[i] = input[v, h]
          indicator.append(v * vmax + h)
          v = v + 1
          i = i + 1
        elif ((v > vmin) and (h < hmax - 1)):  # all other cases
          output[i] = input[v, h]
          indicator.append(v * vmax + h)
          v = v - 1
          h = h + 1
          i = i + 1
      else:  # going down
        if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
          output[i] = input[v, h]
          indicator.append(v * vmax + h)
          h = h + 1
          i = i + 1
        elif (h == hmin):  # if we got to the first column
          output[i] = input[v, h]
          indicator.append(v * vmax + h)
          if (v == vmax - 1):
            h = h + 1
          else:
            v = v + 1
          i = i + 1
        elif ((v < vmax - 1) and (h > hmin)):  # all other cases
          output[i] = input[v, h]
          indicator.append(v * vmax + h)
          v = v + 1
          h = h - 1
          i = i + 1
      if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
        output[i] = input[v, h]
        indicator.append(v * vmax + h)
        break
    return output, indicator

  def inverse_zigzag(self, input, vmax, hmax):
    """
        Zigzag scan of a matrix
        Argument is a two-dimensional matrix of any size,
        not strictly a square one.
        Function returns a 1-by-(m*n) array,
        where m and n are sizes of an input matrix,
        consisting of its items scanned by a zigzag method.
        """
    # initializing the variables
    h = 0
    v = 0

    vmin = 0
    hmin = 0

    output = np.zeros((vmax, hmax))
    i = 0
    while ((v < vmax) and (h < hmax)):
      if ((h + v) % 2) == 0:  # going up
        if (v == vmin):
          output[v, h] = input[i]  # if we got to the first line
          if (h == hmax):
            v = v + 1
          else:
            h = h + 1
          i = i + 1
        elif ((h == hmax - 1) and (v < vmax)):  # if we got to the last column
          output[v, h] = input[i]
          v = v + 1
          i = i + 1
        elif ((v > vmin) and (h < hmax - 1)):  # all other cases
          output[v, h] = input[i]
          v = v - 1
          h = h + 1
          i = i + 1
      else:  # going down
        if ((v == vmax - 1) and (h <= hmax - 1)):  # if we got to the last line
          output[v, h] = input[i]
          h = h + 1
          i = i + 1
        elif (h == hmin):  # if we got to the first column
          output[v, h] = input[i]
          if (v == vmax - 1):
            h = h + 1
          else:
            v = v + 1
          i = i + 1
        elif ((v < vmax - 1) and (h > hmin)):  # all other cases
          output[v, h] = input[i]
          v = v + 1
          h = h - 1
          i = i + 1
      if ((v == vmax - 1) and (h == hmax - 1)):  # bottom right element
        output[v, h] = input[i]
        break
    return output


class SetCriterion(nn.Module):
  """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

  def __init__(self, num_classes, matcher, losses, focal_alpha=0.25):
    """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
    super().__init__()
    self.num_classes = num_classes
    self.matcher = matcher
    self.losses = losses
    self.focal_alpha = focal_alpha
    if 'masks' in losses:
      self.processor_dct = ProcessorDCT(256, 128)

  def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
    assert 'pred_logits' in outputs
    src_logits = outputs['pred_logits']

    idx = self._get_src_permutation_idx(indices)
    target_classes_o = torch.cat(
        [t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2],
                                self.num_classes,
                                dtype=torch.int64,
                                device=src_logits.device)
    target_classes[idx] = target_classes_o

    target_classes_onehot = torch.zeros(
        [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        dtype=src_logits.dtype,
        layout=src_logits.layout,
        device=src_logits.device)
    target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = sigmoid_focal_loss(src_logits,
                                 target_classes_onehot,
                                 num_boxes,
                                 alpha=self.focal_alpha,
                                 gamma=2) * src_logits.shape[1]
    losses = {'loss_ce': loss_ce}

    if log:
      # TODO this should probably be a separate loss, not hacked in this one here
      losses['class_error'] = 100 - accuracy(src_logits[idx],
                                             target_classes_o)[0]
    return losses

  @torch.no_grad()
  def loss_cardinality(self, outputs, targets, indices, num_boxes):
    """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
    pred_logits = outputs['pred_logits']
    device = pred_logits.device
    tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets],
                                  device=device)
    # Count the number of predictions that are NOT "no-object" (which is the last class)
    card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    losses = {'cardinality_error': card_err}
    return losses

  def loss_boxes(self, outputs, targets, indices, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
    assert 'pred_boxes' in outputs
    idx = self._get_src_permutation_idx(indices)
    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat(
        [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(
        generalized_box_iou(box_convert(src_boxes, 'cxcywh', 'xyxy'),
                            box_convert(target_boxes, 'cxcywh', 'xyxy')))
    losses['loss_giou'] = loss_giou.sum() / num_boxes
    return losses

  def loss_centers(self, outputs, targets, indices, num_boxes):
    """Compute the centerness losses related to the bounding boxes, outputs dict must contain the key 'predict centers'
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
    assert 'pred_boxes' in outputs
    idx = self._get_src_permutation_idx(indices)
    src_boxes = outputs['pred_boxes'][idx]
    center_scores = outputs['pred_centers'][idx]
    target_boxes = torch.cat(
        [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    loss_center = centerness_loss(center_scores,
                                  box_convert(src_boxes, 'cxcywh', 'xyxy'),
                                  box_convert(target_boxes, 'cxcywh', 'xyxy'))
    losses = {'loss_center': loss_center.sum() / num_boxes}
    return losses

  def loss_masks(self, outputs, targets, indices, num_boxes):
    """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        pred masks are vectors
        """

    src_idx = self._get_src_permutation_idx(indices)
    tgt_idx = self._get_tgt_permutation_idx(indices)

    src_masks = outputs["pred_masks"]
    src_boxes = outputs['pred_boxes']
    # TODO use valid to mask invalid areas due to padding in loss
    target_boxes = torch.cat(
        [t['xyxy_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    target_masks, valid = nested_tensor_from_tensor_list(
        [t["masks"] for t in targets]).decompose()
    target_masks = target_masks.to(src_masks)
    src_vectors = src_masks[src_idx]
    src_boxes = src_boxes[src_idx]
    target_masks = target_masks[tgt_idx]

    # crop gt_masks
    n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
    gt_masks = BitMasks(target_masks)
    gt_masks = gt_masks.crop_and_resize(
        target_boxes, gt_mask_len).to(device=src_masks.device).float()
    target_masks = gt_masks

    if target_masks.shape[0] == 0:
      losses = {"loss_vector": src_vectors.sum() * 0}
      return losses

    # perform dct transform
    target_vectors = []
    for i in range(target_masks.shape[0]):
      gt_mask_i = ((target_masks[i, :, :] >= 0.5) * 1).to(dtype=torch.uint8)
      gt_mask_i = gt_mask_i.cpu().numpy().astype(np.float32)
      coeffs = cv2.dct(gt_mask_i)
      coeffs = torch.from_numpy(coeffs).flatten()
      coeffs = coeffs[torch.tensor(self.processor_dct.zigzag_table)]
      gt_label = coeffs.unsqueeze(0)
      target_vectors.append(gt_label)

    target_vectors = torch.cat(target_vectors,
                               dim=0).to(device=src_vectors.device)
    losses = {}
    losses['loss_mask'] = F.l1_loss(src_vectors,
                                    target_vectors,
                                    reduction='mean')
    return losses

  def _get_src_permutation_idx(self, indices):
    # permute predictions following indices
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

  def _get_tgt_permutation_idx(self, indices):
    # permute targets following indices
    batch_idx = torch.cat(
        [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx

  def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
    loss_map = {
        'labels': self.loss_labels,
        'cardinality': self.loss_cardinality,
        'boxes': self.loss_boxes,
        'masks': self.loss_masks,
        "centers": self.loss_centers,
    }
    assert loss in loss_map, f'do you really want to compute {loss} loss?'
    return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

  def forward(self, outputs, targets):
    """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

    outputs_without_aux = {
        k: v
        for k, v in outputs.items()
        if k != 'aux_outputs' and k != 'enc_outputs'
    }
    # Retrieve the matching between the outputs of the last layer and the targets
    indices = self.matcher(outputs_without_aux, targets)

    # Compute the average number of target boxes accross all nodes, for normalization purposes
    num_boxes = sum(len(t["labels"]) for t in targets)
    num_boxes = torch.as_tensor([num_boxes],
                                dtype=torch.float,
                                device=next(iter(outputs.values())).device)
    if get_world_size() > 1:
      torch.distributed.all_reduce(num_boxes)
    num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

    # Compute all the requested losses
    losses = {}
    for loss in self.losses:
      kwargs = {}
      losses.update(
          self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

    # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
    if 'aux_outputs' in outputs:
      for i, aux_outputs in enumerate(outputs['aux_outputs']):
        indices = self.matcher(aux_outputs, targets)

        for loss in self.losses:
          if loss == 'masks':
            # Intermediate masks losses are too costly to compute, we ignore them.
            continue
          kwargs = {}
          if loss == 'labels':
            # Logging is enabled only for the last layer
            kwargs['log'] = False
          l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes,
                                 **kwargs)
          l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
          losses.update(l_dict)

    if 'enc_outputs' in outputs:
      enc_outputs = outputs['enc_outputs']
      bin_targets = copy.deepcopy(targets)
      for bt in bin_targets:
        bt['labels'] = torch.zeros_like(bt['labels'])
      indices = self.matcher(enc_outputs, bin_targets)
      for loss in self.losses:
        # if loss == 'masks':
        #   # Intermediate masks losses are too costly to compute, we ignore them.
        #   continue
        kwargs = {}
        if loss == 'labels':
          # Logging is enabled only for the last layer
          kwargs['log'] = False
        l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices,
                               num_boxes, **kwargs)
        l_dict = {k + f'_enc': v for k, v in l_dict.items()}
        losses.update(l_dict)

    return losses
