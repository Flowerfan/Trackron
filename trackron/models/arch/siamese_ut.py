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
from trackron.models.mot import build_mot_head
from trackron.models.objectives import build_objective
from trackron.structures import NestedTensor, ImageList

from .build import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class SiameseUT(nn.Module):

  @configurable
  def __init__(
      self,
      *,
      backbone: nn.Module,
      sot_head: nn.Module,
      mot_head: nn.Module,
      sot_objective: nn.Module,
      mot_objective: nn.Module,
      pixel_mean: Tuple[float],
      pixel_std: Tuple[float]):
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
    self.sot_head = sot_head
    self.mot_head = mot_head
    self.sot_objective = sot_objective
    self.mot_objective = mot_objective
    
    self.register_buffer("pixel_mean",
                         torch.tensor(pixel_mean).view(-1, 1, 1), False)
    self.register_buffer("pixel_std",
                         torch.tensor(pixel_std).view(-1, 1, 1), False)
    assert (self.pixel_mean.shape == self.pixel_std.shape
           ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

  @classmethod
  def from_config(cls, cfg):
    backbone = build_backbone(cfg)
    sot_head = build_sot_head(cfg.MODEL.SOT, backbone._out_feature_channels)
    mot_head = build_mot_head(cfg.MODEL.MOT, backbone._out_feature_channels)
    return {
        "backbone": backbone,
        "sot_head": sot_head,
        "mot_head": mot_head,
        "sot_objective": build_objective(cfg.SOT.OBJECTIVE),
        "mot_objective": build_objective(cfg.MOT.OBJECTIVE),
        "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        "pixel_std": cfg.MODEL.PIXEL_STD,
    }

  def forward(self, data: List[Dict[str, torch.Tensor]], mode='sot'):
    """[summary]

    Args:
        data (List[Dict[str, torch.Tensor]]): [description]
        mode (str, optional): [description]. Defaults to 'sot'.

    Returns:
        [type]: [description]
    """

    samples = self.preprocess_inputs(data)
    # Extract backbone features
    template_features = self.backbone(samples['template'].tensor)
    search_features = self.backbone(samples['search'].tensor)

    if mode == "sot":
      targets = self.prepare_sot_targets(data)
      results = self.sot_head(template_features,
                              targets['template_boxes'],
                              search_features,
                              targets['search_boxes'],
                              template_masks=samples['template'].mask,
                              search_masks=samples['search'].mask)
      targets['search_boxes'] = torch.stack([targets['template_boxes'], targets['search_boxes']], dim=1)
      return self.sot_objective(results, targets)
    elif mode == "mot":
      targets = self.prepare_mot_targets(data)
      results = self.mot_head(template_features, search_features, samples['template'].mask, samples['search'].mask)
      return self.mot_objective(results, targets)


  @property
  def device(self):
    return self.pixel_mean.device

  def preprocess_inputs(self, batched_inputs: List[Dict[str, torch.Tensor]]):
    """
        Normalize, pad and batch the input images.
    """
    samples = {}
    for s in ['template', 'search']:
      images = [
          (x[f"{s}_images"].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs
      ]
      masks = [
          torch.zeros(*x[f'{s}_images'].shape[-2:]).to(self.device)
          for x in batched_inputs
      ]
      # masks = [
      #     x.get(f'{s}_att',
      #           torch.zeros(*x[f'{s}_images'].shape[-2:])).to(self.device)
      #   for x in batched_inputs
      # ]
      samples[s] = ImageList.from_tensors(images, 32, masks=masks)

    ### mask for ori images
    return samples
  
  def prepare_sot_targets(self, data):
    targets = {}
    for s in ['template', 'search']:
      targets[f'{s}_boxes'] = torch.stack([d[f'{s}_boxes'] for d in data]).to(self.device)
      targets[f'{s}_images'] = torch.stack([d[f'{s}_images'] for d in data]).to(self.device)
    return targets


  def prepare_mot_targets(self, data):
    det_targets = []
    track_targets = []
    def norm_boxes(image, boxes):
      H, W = image.shape[-2:]
      NORM_BOX = torch.tensor([W, H, W, H], device=self.device)
      boxes = box_convert(boxes, 'xyxy', 'cxcywh').to(
          self.device) / NORM_BOX
      return boxes
    for idx, d in enumerate(data):
      det_targets += [{
          'boxes': norm_boxes(d['template_images'], d['template_boxes']),
          'labels': d['template_labels'].to(self.device)
      }]
      track_targets += [{
          'boxes': norm_boxes(d['search_images'], d['search_boxes']),
          'labels': d['search_labels'].to(self.device)
      }]
    targets = {'det_targets': det_targets, 'track_targets': track_targets}
    return targets

  def track_sot_init(self, image, box, mask=None):
    image = (image.to(self.device) - self.pixel_mean) / self.pixel_std
    backbone_feat = self.backbone(image)
    return self.sot_head.track_init(backbone_feat, box, mask)
  
  def track_sot(self, image, mask=None, ref_info=None):
    image = (image.to(self.device) - self.pixel_mean) / self.pixel_std
    backbone_feat = self.backbone(image)
    return self.sot_head.track(backbone_feat, mask, ref_info)

  def track_mot(self, image, mask=None, ref_info=None):
    image = (image.to(self.device) - self.pixel_mean[None]) / self.pixel_std[None]
    backbone_feat = self.backbone(image)
    return self.mot_head.track(backbone_feat, mask, ref_info)
