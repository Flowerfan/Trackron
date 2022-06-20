import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from trackron.config import configurable
from trackron.structures import NestedTensor
from trackron.models.box_heads import build_box_head
from trackron.models.extractor import build_extractor
from trackron.models.mask_heads import build_mask_head

from .build import SOT_HEAD_REGISTRY


@SOT_HEAD_REGISTRY.register()
class SiamRPN(nn.Module):

  @configurable
  def __init__(self,
               *,
               feature_layer: list,
               neck: nn.Module,
               rpn_head: nn.Module,
               mask_head: nn.Module,
               refine_head: nn.Module,
               stride: Optional[int] = 16):
    super().__init__()
    self.feature_layer = feature_layer
    self.stride = stride
    self.rpn_head = rpn_head
    self.neck = neck
    self.mask_head = mask_head
    self.refine_head = refine_head

  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    # hidden_dim = cfg.FEATURE_DIM
    neck = None
    if cfg.NECK.ENABLE:
      neck = build_extractor(cfg.NECK)
    rpn_head = build_box_head(cfg.RPN_HEAD)
    mask_head = None
    refine_head = None
    if cfg.MASK_HEAD.ENABLE:
      mask_head = build_mask_head(cfg.MASK_HEAD)
      if cfg.MASK_HEAD.REFINE:
        refine_head = build_mask_head(cfg.MASK_HEAD.REFINE_HEAD)
    return {
        "feature_layer": cfg.FEATURE_LAYER,
        "neck": neck,
        "rpn_head": rpn_head,
        "mask_head": mask_head,
        "refine_head": refine_head,
    }

  def log_softmax(self, cls):
    b, a2, h, w = cls.size()
    cls = cls.view(b, 2, a2 // 2, h, w)
    cls = cls.permute(0, 2, 3, 4, 1).contiguous()
    cls = F.log_softmax(cls, dim=4)
    return cls

  def forward(self,
              template_features,
              template_boxes,
              search_features,
              search_boxes,
              template_masks=None,
              search_masks=None):
    # Extract backbone features
    template_features = [
        template_features[layer] for layer in self.feature_layer
    ]
    search_features = [search_features[layer] for layer in self.feature_layer]
    if self.mask_head is not None:
      template_features = template_features[-1]
      search_features = search_features[-1]
    if self.neck is not None:
      template_features = self.neck(template_features)
      search_features = self.neck(search_features)
    cls_logits, loc_logits = self.rpn_head(template_features, search_features)
    cls_scores = self.log_softmax(cls_logits)

      # get loss
    return {'pred_boxes': loc_logits,
            'pred_scores': cls_scores}

  def get_feature_maskposemb(self, feat, mask):
    C, H, W = feat.shape[1:]
    mask = F.interpolate(mask.float(), size=(H, W)).to(torch.bool).flatten(0, 1)
    nested_feat = NestedTensor(feat, mask)
    pos_emb = self.pos_emb(nested_feat).to(feat.dtype)
    return mask, pos_emb

  def track(self, feature, mask=None, ref_info=None, init_box=None):
    feature = [feature[layer] for layer in self.feature_layer]
    if self.mask_head is not None:
      feature = feature[:-1]
    if self.neck is not None:
      cur_feature = self.neck(feature)
    else:
      cur_feature = feature
    if ref_info is None:
      return {'target_feature': cur_feature}

    target_feature = ref_info['target_feature']
    cls_map, loc_map = self.rpn_head(target_feature, cur_feature)
    out = {"cls": cls_map, "loc": loc_map}
    if self.mask_head is not None:
      mask, self.mask_corr_feature = self.mask_head(target_feature, cur_feature)
      out['mask'] = mask
    return out
