
import torch
import trackron.data.utils as prutils
from torch.nn import functional as F
from torchvision.ops import box_convert
from trackron.config import configurable
from trackron.structures import TensorDict

from ..transforms import transforms as tfm
from .base import SiameseBaseProcessing, stack_tensors
from .build import DATA_PROCESSING_REGISTRY


@DATA_PROCESSING_REGISTRY.register()
class SequenceProcessing(SiameseBaseProcessing):

  @configurable
  def __init__(self,
               search_area_factor,
               output_sz,
               center_jitter_factor,
               scale_jitter_factor,
               crop_type='replicate',
               max_scale_change=None,
               box_mode='xywh',
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.search_area_factor = search_area_factor
    self.output_sz = output_sz
    self.center_jitter_factor = center_jitter_factor
    self.scale_jitter_factor = scale_jitter_factor
    self.crop_type = crop_type
    self.box_mode = box_mode
    self.max_scale_change = max_scale_change

  @classmethod
  def from_config(cls, cfg, training=False):
    search_area_factor = cfg.SEARCH.FACTOR
    output_sz = cfg.SEARCH.SIZE
    center_jitter_factor = cfg.SEARCH.CENTER_JITTER
    scale_jitter_factor = cfg.SEARCH.SCALE_JITTER
    box_mode = cfg.BOX_MODE

    if training:
      transform = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                tfm.ToTensorAndJitter(0.2),
                                tfm.RandomHorizontalFlip(0.5))
    else:
      transform = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                tfm.ToTensorAndJitter(0.2))

    return {
        "search_area_factor": search_area_factor,
        "output_sz": output_sz,
        "center_jitter_factor": center_jitter_factor,
        "scale_jitter_factor": scale_jitter_factor,
        "box_mode": box_mode,
        "transform": transform,
    }

  def _get_jittered_box(self, box):
    jittered_size = box[2:4] * \
        torch.exp(torch.randn(2) * self.scale_jitter_factor)
    max_offset = (jittered_size.prod().sqrt() *
                  torch.tensor(self.center_jitter_factor).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + \
        max_offset * (torch.rand(2) - 0.5)

    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size),
                     dim=0)

  def __call__(self, data: TensorDict):
    # Add a uniform noise to the center pos
    jittered_boxes = [self._get_jittered_box(a) for a in data['search_boxes']]
    crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(
        data['search_images'],
        jittered_boxes,
        data['search_boxes'],
        self.search_area_factor,
        self.output_sz,
        masks=data['search_masks'])

    data['search_images'], data['search_boxes'], data['search_att'], data[
        'search_masks'] = self.transform['search'](image=crops,
                                                   bbox=boxes,
                                                   att=att_mask,
                                                   mask=mask_crops,
                                                   joint=False)

    if data["search_masks"] is None:
      data["search_masks"] = torch.zeros(
          (1, self.output_sz["search"], self.output_sz["search"]))
    # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
    attention_mask = torch.stack(data['search_att'])
    feat_size = self.output_sz // 16  # backbone stride for sot head
    mask = F.interpolate(attention_mask[None].float(),
                         size=feat_size).to(torch.bool)
    if mask.flatten(-2).all(-1).any():
      raise ValueError('mask wrong, will get nan loss nan')

    # Prepare output
    data = data.apply(stack_tensors)
    if self.box_mode == 'xyxy':
      data['search_boxes'] = box_convert(data['search_boxes'], 'xywh', 'xyxy')

    return data
