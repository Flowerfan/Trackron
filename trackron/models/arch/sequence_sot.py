import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from trackron.config import configurable
from trackron.models.backbone import build_backbone
from trackron.models.sot import build_sot_head
from trackron.models.objectives import build_objective

from .build import META_ARCH_REGISTRY, BaseModel

@META_ARCH_REGISTRY.register()
class SequenceSOT(BaseModel):
  """Example are sequence of a video
  """

  @configurable
  def __init__(
      self,
      *,
      backbone: nn.Module,
      track_head: nn.Module,
      objective: nn.Module,
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
        "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        "pixel_std": cfg.MODEL.PIXEL_STD,
    }

  def forward_sot(self, data: Dict[str, torch.Tensor]):
    images, masks = self.preprocess_inputs(data)
    boxes = data['search_boxes'].to(self.device)
    # Extract backbone features
    features = self.backbone(images.flatten(0, 1))

    results = self.track_head(features, boxes, masks=masks)
    
    return self.objective(results, data)

  @property
  def device(self):
    return self.pixel_mean.device
  
  def preprocess_inputs(self, inputs: Dict[str, torch.Tensor]):
    """

    Args:
        inputs (Dict[str, torch.Tensor]): [description]

    Returns:
        images: [type]: [description]
        masks: image masks
    """
    images = (inputs['search_images'].to(self.device) - self.pixel_mean[None]) / self.pixel_std[None]
    ### mask for ori images
    masks = inputs.get('search_att', torch.zeros(*images.shape[:2], *images.shape[-2:])).to(self.device)
    return images, masks
  
  def track_sot(self, image, mask=None, ref_info=None, init_box=None):
    assert ref_info is None or init_box is None
    image = (image.to(self.device) - self.pixel_mean) / self.pixel_std
    backbone_feat = self.backbone(image)
    return self.track_head.track(backbone_feat, mask, ref_info, init_box=init_box)
    
