import math
import torch
import numpy as np
from torchvision.ops import box_convert
import trackron.data.utils as prutils
from trackron.config import configurable
# from evaluation.utils.visdom import Visdom
from trackron.structures import TensorDict, Anchors, corner2center

from ..transforms import transforms as tfm
from .base import SiameseBaseProcessing
from .build import DATA_PROCESSING_REGISTRY


@DATA_PROCESSING_REGISTRY.register()
class SiamRPNProcessing(SiameseBaseProcessing):

  @configurable
  def __init__(self,
               img_sz,
               out_sz,
               search_area_factor,
               anchors,
               center_jitter_factor,
               scale_jitter_factor,
               crop_type='replicate',
               max_scale_change=None,
               label_function_params=None,
               *args,
               **kwargs):
    """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
            max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
    super().__init__(*args, **kwargs)
    self.img_sz = img_sz
    self.out_sz = out_sz
    self.search_area_factor = search_area_factor
    self.center_jitter_factor = center_jitter_factor
    self.scale_jitter_factor = scale_jitter_factor
    self.crop_type = crop_type
    self.max_scale_change = max_scale_change
    self.label_function_params = label_function_params
    self.anchors = anchors

    # self.visdom = Visdom(1, None, {})

  @classmethod
  def from_config(cls, cfg, training=False):
    search_area_factor = cfg.SEARCH.FACTOR
    img_sz = {'template': cfg.TEMPLATE.SIZE, 'search': cfg.SEARCH.SIZE}
    out_sz = {'template': cfg.TEMPLATE.OUT_SIZE, 'search': cfg.SEARCH.OUT_SIZE}
    center_jitter_factor = {
        'template': cfg.TEMPLATE.CENTER_JITTER,
        'search': cfg.SEARCH.CENTER_JITTER
    }
    scale_jitter_factor = {
        'template': cfg.TEMPLATE.SCALE_JITTER,
        'search': cfg.SEARCH.SCALE_JITTER
    }
    anchors = Anchors(cfg.ANCHOR.STRIDE, cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)
    anchors.generate_all_anchors(im_c=cfg.SEARCH.SIZE // 2,
                                 size=cfg.SEARCH.OUT_SIZE)

    if training:
      transform = tfm.Transform(tfm.ToTensorAndJitter(0.2, normalize=False),
                                tfm.RandomHorizontalFlip(0.5))
      transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                      tfm.RandomHorizontalFlip(0.5))
    else:
      transform = tfm.Transform(tfm.ToTensorAndJitter(0.2, normalize=False))
      transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    return {
        "img_sz": img_sz,
        "out_sz": out_sz,
        "search_area_factor": search_area_factor,
        "anchors": anchors,
        "center_jitter_factor": center_jitter_factor,
        "scale_jitter_factor": scale_jitter_factor,
        "transform": transform,
        "joint_transform": transform_joint,
    }

  def _get_jittered_box(self, box, mode):
    """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

    jittered_size = box[2:4] * \
        torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
    max_offset = (jittered_size.prod().sqrt() *
                  torch.tensor(self.center_jitter_factor[mode]).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + \
        max_offset * (torch.rand(2) - 0.5)

    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size),
                     dim=0)

  def _generate_labels(self, target, size, neg=False):
    anchor_num = len(self.anchors.ratios) * len(self.anchors.scales)

    # -1 ignore 0 negative 1 positive
    cls = -1 * torch.ones((anchor_num, size, size), dtype=torch.long)
    delta = torch.zeros((4, anchor_num, size, size), dtype=torch.float32)
    delta_weight = torch.zeros((anchor_num, size, size), dtype=torch.float32)

    def select(position, keep_num=16):
      num = position[0].shape[0]
      if num <= keep_num:
        return position, num
      slt = torch.randperm(num)
      slt = slt[:keep_num]
      return tuple(p[slt] for p in position), keep_num

    tcx, tcy, tw, th = box_convert(target, 'xyxy', 'cxcywh')

    if neg:
      # l = size // 2 - 3
      # r = size // 2 + 3 + 1
      # cls[:, l:r, l:r] = 0

      cx = size // 2
      cy = size // 2
      cx += torch.ceil((tcx - self.img_sz['search'] // 2) / self.anchors.stride + 0.5)
      cy += torch.ceil((tcy - self.img_sz['search'] // 2) / self.anchors.stride + 0.5)
      l = max(0, cx - 3)
      r = min(size, cx + 4)
      u = max(0, cy - 3)
      d = min(size, cy + 4)
      cls[:, u:d, l:r] = 0

      neg, neg_num = select((cls == 0).nonzero(as_tuple=True), 16)
      cls[:] = -1
      cls[neg] = 0

      overlap = torch.zeros((anchor_num, size, size), dtype=torch.float32)
      return cls, delta, delta_weight, overlap

    anchor_box = torch.tensor(self.anchors.all_anchors[0])
    anchor_center = torch.tensor(self.anchors.all_anchors[1])
    x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
        anchor_box[2], anchor_box[3]
    cx, cy, w, h = anchor_center[0], anchor_center[1], \
        anchor_center[2], anchor_center[3]

    delta[0] = (tcx - cx) / w
    delta[1] = (tcy - cy) / h
    delta[2] = torch.log(tw / w)
    delta[3] = torch.log(th / h)

    overlap = prutils.iou_all(target, torch.stack([x1, y1, x2, y2], dim=-1))

    pos = (overlap > 0.6).nonzero(as_tuple=True)
    neg = (overlap < 0.3).nonzero(as_tuple=True)

    pos, pos_num = select(pos, 16)
    neg, neg_num = select(neg, 48)

    cls[pos] = 1
    delta_weight[pos] = 1. / (pos_num + 1e-6)

    cls[neg] = 0
    return cls, delta, delta_weight, overlap

  def __call__(self, data: TensorDict):
    """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_boxes', 'search_boxes'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_boxes', 'search_boxes', 'search_proposals', 'proposal_iou',
                'search_label' (optional), 'template_label' (optional), 'search_label_density' (optional), 'template_label_density' (optional)
        """

    if self.transform['joint'] is not None:
      data['template_images'], data['template_boxes'] = self.transform['joint'](
          image=data['template_images'], bbox=data['template_boxes'])
      data['search_images'], data['search_boxes'] = self.transform['joint'](
          image=data['search_images'],
          bbox=data['search_boxes'],
          new_roll=False)

    for s in ['template', 'search']:
      if s + '_masks' in data:
        data.pop(s + '_masks')

      # Add a uniform noise to the center pos
      jittered_boxes = [
          self._get_jittered_box(a, s) for a in data[s + '_boxes']
      ]

      crops, boxes, _ = prutils.target_image_crop(
          data[s + '_images'],
          jittered_boxes,
          data[s + '_boxes'],
          self.search_area_factor,
          self.img_sz[s],
          mode=self.crop_type,
          max_scale_change=self.max_scale_change)
      # self.visdom.register((crops[0], boxes[0]), 'Tracking', 1, 'Tracking')

      data[s + '_images'], data[s + '_boxes'] = self.transform[s](image=crops,
                                                                  bbox=boxes,
                                                                  joint=False)

    # Prepare output
    data = data.apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
    data['search_images'] = data['search_images'][[2, 1, 0], :, :] ### BGR for siamrpn
    data['template_images'] = data['template_images'][[2, 1, 0], :, :] ### BGR for siamrpn
    data['search_boxes'] = box_convert(data['search_boxes'], 'xywh', 'xyxy')

    # Generate labels
    cls, delta, delta_weight, overlap = self._generate_labels(
        data['search_boxes'], self.out_sz['search'])
    data['search_label'] = cls
    data['search_boxes_delta'] = delta
    data['search_boxes_delta_weight'] = delta_weight
    return data
