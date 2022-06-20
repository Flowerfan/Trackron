import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision.ops import box_convert
from einops import rearrange, reduce
from timm.models.layers import DropPath
from trackron.config import configurable
from trackron.models.poolers import ROIPooler
from trackron.models.layers.activation import get_activation_fn
from trackron.models.layers.blocks import conv_block
from trackron.utils.misc import clone_modules
from trackron.structures import Boxes

from .build import MASK_HEAD_REGISTRY


@MASK_HEAD_REGISTRY.register()
class DynamicMaskHead2(nn.Module):

  @configurable
  def __init__(
      self,
      iterations,
      pooler,
      interact_layers,
      box_module,
      score_module,
      box_transform,
      output_score,
  ):
    super().__init__()

    self.iterations = iterations
    self.interact_layers = interact_layers
    self.pooler = pooler
    self.box_module = box_module
    self.score_module = score_module
    self.box_transform = box_transform
    self.output_score = output_score

  @classmethod
  def from_config(cls, cfg):
    iterations = cfg.ITERATIONS
    hidden_dim = cfg.FEATURE_DIM
    dynamic_dim = cfg.DYNAMIC_DIM
    num_dynamic = cfg.NUM_DYNAMIC
    dim_feedforward = cfg.DIM_FEEDFORWARD
    dropout = cfg.DROPOUT
    pooler_resolution = cfg.POOL_SIZE
    pooler_scales = cfg.POOL_SCALES
    sampling_ratio = cfg.POOL_SAMPLE_RATIO
    pooler_type = cfg.POOL_TYPE
    activation = cfg.ACTIVATION
    num_box_layer = cfg.NUM_BOX_LAYER
    box_weights = cfg.BOX_WEIGHTS
    output_score = cfg.OUTPUT_SCORE

    pooler = ROIPooler(
        output_size=pooler_resolution,
        scales=pooler_scales,
        sampling_ratio=sampling_ratio,
        pooler_type=pooler_type,
    )
    interact = DynamicConv(hidden_dim, dynamic_dim, num_dynamic,
                           pooler_resolution)
    interact_layer = InteractLayer(hidden_dim,
                                   interact,
                                   dim_feedforward,
                                   dropout=dropout,
                                   activation=activation)
    box_layer = nn.Sequential(
        *[
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim, False),
                          nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True))
            for _ in range(num_box_layer)
        ], nn.Linear(hidden_dim, 4))
    interact_layers = clone_modules(interact_layer, iterations)
    box_module = clone_modules(box_layer, iterations)
    box_transform = Box2BoxTransform(box_weights)
    score_module = None
    if output_score:
      score_layer = nn.Sequential(
          *[
              nn.Sequential(nn.Linear(hidden_dim, hidden_dim, False),
                            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True))
              for _ in range(num_box_layer)
          ], nn.Linear(hidden_dim, 1))
      score_module = clone_modules(score_layer, iterations)
    return {
        "iterations": iterations,
        "interact_layers": interact_layers,
        "pooler": pooler,
        "box_module": box_module,
        "score_module": score_module,
        "box_transform": box_transform,
        "output_score": output_score,
    }

  def forward(self,
              features,
              obj_features,
              boxes,
              intermediate=True,
              pooler=None):
    """[summary]

    Args:
        features ([torch.Tensor]): [B C H W]
        obj_features ([torch.Tensor]): [B N C]
        boxes ([torch.Tensor]): [B N 4]

    Returns:
        [type]: [description]
    """

    B, N = boxes.shape[:2]
    outputs = list()
    scores = list()
    obj_embs = list()
    pooler = pooler if pooler is not None else self.pooler
    if not isinstance(features, (list, tuple)):
      features = [features]
    for it in range(self.iterations):
      # roi_feature.
      proposal_boxes = list()
      for b in range(B):
        proposal_boxes.append(Boxes(boxes[b]))
      roi_features = pooler(features, proposal_boxes)
      roi_features = rearrange(roi_features, 'BN C H W -> BN (H W) C')

      ### refine obj_feature
      # box_feature = obj_features.transpose(0, 1).reshape(B * N, -1)
      obj_features = self.interact_layers[it](roi_features, obj_features)
      boxes_deltas = self.box_module[it](obj_features)
      boxes = box_convert(boxes.view(-1, 4), 'xyxy', 'xywh')
      pred_bboxes = self.box_transform.apply_deltas(boxes_deltas.view(-1, 4),
                                                    boxes)
      boxes = box_convert(pred_bboxes, 'xywh', 'xyxy').view(B, N, 4)
      outputs.append(boxes)
      obj_embs.append(obj_features)
      if self.output_score:
        scores.append(self.score_module[it](obj_features).squeeze(-1))
    if intermediate:
      return outputs, scores, obj_embs
    return outputs[-1], scores[-1], obj_embs[-1]


class InteractLayer(nn.Module):

  def __init__(
      self,
      d_model,
      interact,
      dim_feedforward=2048,
      nhead=8,
      dropout=0.1,
      activation="relu",
  ):
    super().__init__()

    self.d_model = d_model

    # dynamic.
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.interact = interact

    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    self.activation = get_activation_fn(activation)

  def forward(self, roi_features, obj_features):
    """[summary]

    Args:
        roi_features ([tensor]): [BN K C]
        obj_features ([tensor]): [B N C]

    Returns:
        [type]: [description]
    """
    input_shape = obj_features.shape
    # proposal features
    obj_features = rearrange(obj_features, 'B N C -> N B C')
    obj_features2 = self.self_attn(obj_features,
                                   obj_features,
                                   value=obj_features)[0]
    obj_features = obj_features + self.dropout1(obj_features2)
    obj_features = self.norm1(obj_features)

    ### interact
    obj_features = rearrange(obj_features, 'N B C -> (B N) C')
    obj_features2 = self.interact(obj_features.unsqueeze(1), roi_features)
    obj_features = obj_features + self.dropout2(obj_features2)
    obj_features = self.norm2(obj_features)

    # obj_feature.
    obj_features2 = self.linear2(
        self.dropout(self.activation(self.linear1(obj_features))))
    obj_features = obj_features + self.dropout3(obj_features2)
    obj_features = self.norm3(obj_features)

    return obj_features.reshape(*input_shape)


class DynamicConv(nn.Module):

  def __init__(self, hidden_dim, dynamic_dim, num_dynamic, pooler_resolution):
    super().__init__()

    self.hidden_dim = hidden_dim
    self.dim_dynamic = dynamic_dim
    self.num_dynamic = num_dynamic
    self.num_params = self.hidden_dim * self.dim_dynamic
    self.dynamic_layer = nn.Linear(self.hidden_dim,
                                   self.num_dynamic * self.num_params)

    self.norm1 = nn.LayerNorm(self.dim_dynamic)
    self.norm2 = nn.LayerNorm(self.hidden_dim)

    self.activation = nn.ReLU(inplace=True)

    num_output = self.hidden_dim * pooler_resolution**2
    self.out_layer = nn.Linear(num_output, self.hidden_dim)
    self.norm3 = nn.LayerNorm(self.hidden_dim)

  def forward(self, pro_features, roi_features):
    '''
        pro_features: (N, 1, C)
        roi_features: (N, 49, C)
        '''
    features = roi_features
    parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

    param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim,
                                                     self.dim_dynamic)
    param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic,
                                                     self.hidden_dim)

    features = torch.bmm(features, param1)
    features = self.norm1(features)
    features = self.activation(features)

    features = torch.bmm(features, param2)
    features = self.norm2(features)
    features = self.activation(features)

    features = features.flatten(1)
    features = self.out_layer(features)
    features = self.norm3(features)
    features = self.activation(features)

    return features


@MASK_HEAD_REGISTRY.register()
class DynamicMaskHead(nn.Module):

  @configurable
  def __init__(self, input_dim, hidden_dim, upsample):
    super().__init__()

    self.conv1 = conv_block(input_dim, hidden_dim)
    self.conv2 = conv_block(hidden_dim, hidden_dim // 2)
    self.conv3 = conv_block(hidden_dim // 2, hidden_dim // 4)
    self.conv4 = conv_block(hidden_dim // 4, hidden_dim // 8)
    self.conv_logits = nn.Conv2d(hidden_dim // 8, 1, 1)
    self.upsample = upsample

  @classmethod
  def from_config(cls, cfg):
    upsample = None
    hidden_dim = cfg.HIDDEN_DIM
    if cfg.UPSAMPLE:
      upsample = nn.Sequential(
          nn.ConvTranspose2d(hidden_dim // 8,
                             hidden_dim // 8,
                             cfg.SCALE_FACTOR,
                             stride=cfg.SCALE_FACTOR), nn.ReLU())
    return {
        "input_dim": cfg.INPUT_DIM,
        "hidden_dim": hidden_dim,
        "upsample": upsample
    }

  def forward(self, tgt_features, roi_features):
    '''
        tgt_features: (N, C)
        roi_features: (N, K, H, W) K = H*W (H=W)
    return target_mask: (N C H W)
        '''
    height, width = roi_features.shape[-2:]
    flat_roi = roi_features.flatten(-2)
    mask = torch.einsum('nc,nck->nk', tgt_features, flat_roi)
    target_mask = mask.unsqueeze(1) * flat_roi
    target_mask = rearrange(target_mask, 'n c (h w) -> n c h w', h=height, w=width)
    target_mask = self.conv1(target_mask)
    target_mask = self.conv2(target_mask)
    target_mask = self.conv3(target_mask)
    target_mask = self.conv4(target_mask)
    if self.upsample is not None:
      target_mask = self.upsample(target_mask)

    target_mask = self.conv_logits(target_mask)
    return target_mask
