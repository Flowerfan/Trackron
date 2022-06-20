import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from torchvision.ops import box_convert

from trackron.config import configurable
from trackron.models.backbone import build_backbone
from trackron.models.objectives import build_objective
from trackron.structures import ImageList

from .build import META_ARCH_REGISTRY

@META_ARCH_REGISTRY.register()
class DetectionMOT(nn.Module):

  @configurable
  def __init__(
      self,
      *,
      model: nn.Module,
      objective: nn.Module,
      pixel_mean: Tuple[float],
      pixel_std: Tuple[float]):
    """[summary]

    Args:
        classification_layers (list): [description]
        output_layers (list): [description]
        backbone (nn.Module): [description]
        pixel_mean (Tuple[float]): [description]
        pixel_std (Tuple[float]): [description]
    """
    super().__init__()
    self.model = model
    self.objective = objective
    self.register_buffer("pixel_mean",
                         torch.tensor(pixel_mean).view(-1, 1, 1), False)
    self.register_buffer("pixel_std",
                         torch.tensor(pixel_std).view(-1, 1, 1), False)
    assert (self.pixel_mean.shape == self.pixel_std.shape
           ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

  @classmethod
  def from_config(cls, cfg):
    model = build_backbone(cfg)
    return {
        "model": model,
        "objective": build_objective(cfg.MOT.OBJECTIVE),
        "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        "pixel_std": cfg.MODEL.PIXEL_STD,
    }

  def forward(self, data: Dict[str, torch.Tensor]):
    # if not self.training:
    # def forward(self, template_imgs, search_imgs, template_bb, search_proposals, *args, **kwargs):
    samples = self.preprocess_inputs(data)
    targets = self.prepare_mot_targets(data)
    # Extract backbone features
    template_features = self.backbone(samples['template'].tensor)
    search_features = self.backbone(samples['search'].tensor)

    results = self.track_head(template_features, search_features, samples['template'].mask, samples['search'].mask)

    return self.objective(results, targets)

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
          x.get(f'{s}_att',
                torch.zeros(*x[f'{s}_images'].shape[-2:])).to(self.device)
        for x in batched_inputs
      ]
      samples[s] = ImageList.from_tensors(images, 32, masks=masks)

    ### mask for ori images
    return samples

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

  def track_mot(self, image, mask=None, ref_info=None):
    image = (image.to(self.device) - self.pixel_mean[None]) / self.pixel_std[None]
    outputs = self.model(image)
    return outputs
