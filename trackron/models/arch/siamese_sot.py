import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from einops import rearrange, reduce, repeat
from torchvision.ops import box_convert

from trackron.config import configurable
from trackron.models.backbone import build_backbone
from trackron.models.sot import build_sot_head
from trackron.models.objectives import build_objective
from trackron.structures import NestedTensor, ImageList

from .build import META_ARCH_REGISTRY, BaseModel


@META_ARCH_REGISTRY.register()
class SiameseSOT(BaseModel):

  @configurable
  def __init__(self,
               *,
               backbone: nn.Module,
               track_head: nn.Module,
               objective: nn.Module,
               stack_objective: bool,
               pixel_mean: Tuple[float],
               pixel_std: Tuple[float],
               stride: Optional[int] = 16):
    """[summary]

    Args:
        classification_layers (list): [description]
        output_layers (list): [description]
        backbone (nn.Module): [description]
        cls_head (nn.Module): [description]
        box_head (nn.Module): [description]
        pixel_mean (Tuple[float]): [description]
        pixel_std (Tuple[float]): [description]
    """
    super().__init__()
    self.backbone = backbone
    self.track_head = track_head
    self.objective = objective
    self.stack_objective = stack_objective
    self.register_buffer("pixel_mean",
                         torch.tensor(pixel_mean).view(1, -1, 1, 1), False)
    self.register_buffer("pixel_std",
                         torch.tensor(pixel_std).view(1, -1, 1, 1), False)
    assert (self.pixel_mean.shape == self.pixel_std.shape
           ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

  @classmethod
  def from_config(cls, cfg):
    backbone = build_backbone(cfg)
    track_head = build_sot_head(cfg.MODEL.SOT, backbone._out_feature_channels)
    return {
        "backbone": backbone,
        "track_head": track_head,
        "objective": build_objective(cfg.SOT.OBJECTIVE),
        "stack_objective": cfg.SOT.OBJECTIVE.STACK,
        "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        "pixel_std": cfg.MODEL.PIXEL_STD,
    }

  def forward_sot(self, data: Dict[str, torch.Tensor]):
    samples = self.preprocess_inputs(data)
    template_boxes = data['template_boxes'].to(self.device)
    search_boxes = data['search_boxes'].to(self.device)
    # Extract backbone features
    template_features = self.backbone(samples['template_images'])
    search_features = self.backbone(samples['search_images'])

    results = self.track_head(template_features,
                              template_boxes,
                              search_features,
                              search_boxes,
                              template_masks=samples['template_masks'],
                              search_masks=samples['search_masks'])
    if self.stack_objective:
      data['search_boxes'] = torch.stack([template_boxes, search_boxes], dim=1)

    return self.objective(results, data)

  @property
  def device(self):
    return self.pixel_mean.device

  def preprocess_inputs(self, inputs: Dict[str, torch.Tensor]):
    """
        Normalize, pad and batch the input images.
    """
    samples = {}
    for s in ['template', 'search']:
      shape = list(inputs[f'{s}_images'].shape)
      if len(shape) == 4:
        images = (inputs[f'{s}_images'].to(self.device) -
                  self.pixel_mean) / self.pixel_std
        masks = inputs.get(f'{s}_att', torch.zeros(shape[0], shape[2],
                                                   shape[3])).to(self.device)
      elif len(shape) == 5:
        images = (inputs[f'{s}_images'].to(self.device) -
                  self.pixel_mean[None]) / self.pixel_std[None]
        masks = inputs.get(f'{s}_att',
                           torch.zeros(shape[0], shape[1], shape[3],
                                       shape[4])).to(self.device)
      samples[f'{s}_images'] = images
      samples[f'{s}_masks'] = masks
    ### mask for ori images
    return samples

  def track_sot(self, image, mask=None, ref_info=None, init_box=None):
    assert ref_info is None or init_box is None
    image = (image.to(self.device) - self.pixel_mean) / self.pixel_std
    backbone_feat = self.backbone(image)
    return self.track_head.track(backbone_feat, mask, ref_info, init_box=init_box)