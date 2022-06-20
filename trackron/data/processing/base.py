import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from trackron.structures import TensorDict
from torchvision.ops import box_convert
import trackron.data.utils as prutils
from .build import DATA_PROCESSING_REGISTRY
from torch.jit.annotations import Optional, List, Dict, Tuple
from trackron.utils.visdom import Visdom
from trackron.config import configurable
from ..transforms import transforms as tfm


def stack_tensors(x):
  if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
    return torch.stack(x)
  return x


class SiameseBaseProcessing:
  """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

  def __init__(self,
               transform=transforms.ToTensor(),
               template_transform=None,
               search_transform=None,
               joint_transform=None):
    """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both search and template images to grayscale.
        """
    self.transform = {
        'template':
            transform if template_transform is None else template_transform,
        'search':
            transform if search_transform is None else search_transform,
        'joint':
            joint_transform
    }

  def __call__(self, data: TensorDict):
    raise NotImplementedError


@DATA_PROCESSING_REGISTRY.register()
class SiamProcessing(SiameseBaseProcessing):
  """ The processing class used for training DiMP. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
    used for computing the loss of the predicted classification model on the search images. A set of proposals are
    also generated for the search images by jittering the ground truth box. These proposals are used to template the
    bounding box estimating branch.

    """

  @configurable
  def __init__(self,
               search_area_factor,
               output_sz,
               center_jitter_factor,
               scale_jitter_factor,
               crop_type='replicate',
               max_scale_change=None,
               mode='pair',
               box_mode='xywh',
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
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
    super().__init__(*args, **kwargs)
    self.search_area_factor = search_area_factor
    self.output_sz = output_sz
    self.center_jitter_factor = center_jitter_factor
    self.scale_jitter_factor = scale_jitter_factor
    self.crop_type = crop_type
    self.mode = mode
    self.box_mode = box_mode
    self.max_scale_change = max_scale_change

    # self.visdom = Visdom(2, None, {})

  @classmethod
  def from_config(cls, cfg, training=False):
    search_area_factor = cfg.SEARCH.FACTOR
    output_sz = cfg.SEARCH.SIZE
    center_jitter_factor = {
        'template': cfg.TEMPLATE.CENTER_JITTER,
        'search': cfg.SEARCH.CENTER_JITTER
    }
    scale_jitter_factor = {
        'template': cfg.TEMPLATE.SCALE_JITTER,
        'search': cfg.SEARCH.SCALE_JITTER
    }
    mode = cfg.TRAIN.MODE
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
        "mode": mode,
        "box_mode": box_mode,
        "transform": transform,
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

    for s in ['template', 'search']:
      assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
          "In pair mode, num template/search frames must be 1"

      # Add a uniform noise to the center pos
      jittered_boxes = [
          self._get_jittered_box(a, s) for a in data[s + '_boxes']
      ]

      crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(
          data[s + '_images'],
          jittered_boxes,
          data[s + '_boxes'],
          self.search_area_factor,
          self.output_sz,
          masks=data[s + '_masks'])

      data[s + '_images'], data[s + '_boxes'], data[s + '_att'], data[
          s + '_masks'] = self.transform[s](image=crops,
                                            bbox=boxes,
                                            att=att_mask,
                                            mask=mask_crops,
                                            joint=False)

      if data[f"{s}_masks"] is None:
        data[f"{s}_masks"] = torch.zeros(
            (1, self.output_sz[s], self.output_sz[s]))

      # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
      image_mask = torch.stack(data[s + '_att'])
      feat_size = self.output_sz // 16  # backbone stride for sot head
      mask = F.interpolate(image_mask[None].float(), size=feat_size).to(torch.bool)
      if mask.flatten(-2).all(-1).any():
        raise ValueError('mask wrong, will get nan loss nan')
    # self.visdom.register(((data['template_images'][0].permute(1,2,0)*255).numpy().astype('uint8'), data['template_boxes'][0]), 'Tracking', 1, 'Template')
    # self.visdom.register(((data['search_images'][0].permute(1,2,0)*255).numpy().astype('uint8'), data['search_boxes'][0]), 'Tracking', 1, 'Search')
    # Prepare output
    if self.mode == 'sequence':
      data = data.apply(stack_tensors)
    else:
      data = data.apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
    if self.box_mode == 'xyxy':
      data['search_boxes'] = box_convert(data['search_boxes'], 'xywh', 'xyxy')
      data['template_boxes'] = box_convert(data['template_boxes'], 'xywh',
                                           'xyxy')

    return data

@DATA_PROCESSING_REGISTRY.register()
class SiamProcessingBRPadding(SiamProcessing):
  """ The processing class used for training DiMP. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
    used for computing the loss of the predicted classification model on the search images. A set of proposals are
    also generated for the search images by jittering the ground truth box. These proposals are used to template the
    bounding box estimating branch.

    """

  @classmethod
  def from_config(cls, cfg, training=False):
    search_area_factor = cfg.SEARCH.FACTOR
    output_sz = cfg.SEARCH.SIZE
    center_jitter_factor = {
        'template': cfg.TEMPLATE.CENTER_JITTER,
        'search': cfg.SEARCH.CENTER_JITTER
    }
    scale_jitter_factor = {
        'template': cfg.TEMPLATE.SCALE_JITTER,
        'search': cfg.SEARCH.SCALE_JITTER
    }
    mode = cfg.TRAIN.MODE
    box_mode = cfg.BOX_MODE

    if training:
      transform = tfm.Transform(tfm.ToTensorAndJitter(0.2))
      transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                      tfm.RandomHorizontalFlip(0.5))
    else:
      transform = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                tfm.ToTensorAndJitter(0.2))

    return {
        "search_area_factor": search_area_factor,
        "output_sz": output_sz,
        "center_jitter_factor": center_jitter_factor,
        "scale_jitter_factor": scale_jitter_factor,
        "mode": mode,
        "box_mode": box_mode,
        "transform": transform,
        "joint_transform": transform_joint,
    }

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
      data['template_images'], data['template_boxes'], data['template_masks'] = self.transform['joint'](
          image=data['template_images'], bbox=data['template_boxes'], mask=data['template_masks'])
      data['search_images'], data['search_boxes'], data['search_masks'] = self.transform['joint'](
          image=data['search_images'], bbox=data['search_boxes'], mask=data['search_masks'], new_roll=False)

    for s in ['template', 'search']:
      assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
          "In pair mode, num template/search frames must be 1"

      # Add a uniform noise to the center pos
      jittered_boxes = [
          self._get_jittered_box(a, s) for a in data[s + '_boxes']
      ]

      crops, boxes, att_mask, mask_crops = prutils.jittered_brpadding_crop(
          data[s + '_images'],
          jittered_boxes,
          data[s + '_boxes'],
          self.search_area_factor,
          self.output_sz,
          masks=data[s + '_masks'])

      data[s + '_images'], data[s + '_boxes'], data[s + '_att'], data[
          s + '_masks'] = self.transform[s](image=crops,
                                            bbox=boxes,
                                            att=att_mask,
                                            mask=mask_crops,
                                            joint=False)

      if data[f"{s}_masks"] is None:
        data[f"{s}_masks"] = torch.zeros(
            (1, self.output_sz[s], self.output_sz[s]))

      # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
      image_mask = torch.stack(data[s + '_att'])
      feat_size = self.output_sz // 32  # backbone stride for sot head
      mask = F.interpolate(image_mask[None].float(), size=feat_size).to(torch.bool)
      if mask.flatten(-2).all(-1).any():
        raise ValueError('mask wrong, will get nan loss nan')
    # if getattr(self, 'visdom', None) is None:
    #   self.visdom = Visdom(2, None, {})
    # self.visdom.register(((data['template_images'][0].permute(1,2,0)*255).numpy().astype('uint8'), data['template_boxes'][0]), 'Tracking', 1, 'Template')
    # self.visdom.register(((data['search_images'][0].permute(1,2,0)*255).numpy().astype('uint8'), data['search_boxes'][0]), 'Tracking', 1, 'Search')
    # Prepare output
    if self.mode == 'sequence':
      data = data.apply(stack_tensors)
    else:
      data = data.apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
    if self.box_mode == 'xyxy':
      data['search_boxes'] = box_convert(data['search_boxes'], 'xywh', 'xyxy')
      data['template_boxes'] = box_convert(data['template_boxes'], 'xywh',
                                           'xyxy')

    return data
