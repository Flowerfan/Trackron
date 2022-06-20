import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from trackron import config
from trackron.structures import TensorDict
from torchvision.ops import box_convert
import trackron.data.utils as prutils
from .build import DATA_PROCESSING_REGISTRY
from torch.jit.annotations import Optional, List, Dict, Tuple
from trackron.utils.visdom import Visdom
from trackron.config import configurable
from ..transforms import transforms as tfm
from .base import SiameseBaseProcessing


def stack_tensors(x):
  if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
    return torch.stack(x)
  return x


@DATA_PROCESSING_REGISTRY.register()
class VOSProcessing(SiameseBaseProcessing):
  """ The processing class used for training LWL. The images are processed in the following way.
    First, the target bounding box (computed using the segmentation mask)is jittered by adding some noise.
    Next, a rectangular region (called search region ) centered at the jittered target center, and of area
    search_area_factor^2 times the area of the jittered box is cropped from the image.
    The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. The argument 'crop_type' determines how out-of-frame regions are handled when cropping the
    search region. For instance, if crop_type == 'replicate', the boundary pixels are replicated in case the search
    region crop goes out of frame. If crop_type == 'inside_major', the search region crop is shifted/shrunk to fit
    completely inside one axis of the image.
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
               new_roll=False,
               *args,
               **kwargs):
    """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - The size (width, height) to which the search region is resized. The aspect ratio is always
                        preserved when resizing the search region
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - Determines how out-of-frame regions are handled when cropping the search region.
                        If 'replicate', the boundary pixels are replicated in case the search region crop goes out of
                                        image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis
                        of the image.
            max_scale_change - Maximum allowed scale change when shrinking the search region to fit the image
                               (only applicable to 'inside' and 'inside_major' cropping modes). In case the desired
                               shrink factor exceeds the max_scale_change, the search region is only shrunk to the
                               factor max_scale_change. Out-of-frame regions are then handled by replicating the
                               boundary pixels. If max_scale_change is set to None, unbounded shrinking is allowed.

            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            new_roll - Whether to use the same random roll values for train and test frames when applying the joint
                       transformation. If True, a new random roll is performed for the test frame transformations. Thus,
                       if performing random flips, the set of train frames and the set of test frames will be flipped
                       independently.
        """
    super().__init__(*args, **kwargs)
    self.search_area_factor = search_area_factor
    self.output_sz = output_sz
    self.center_jitter_factor = center_jitter_factor
    self.scale_jitter_factor = scale_jitter_factor
    self.crop_type = crop_type
    self.mode = mode
    self.max_scale_change = max_scale_change

    self.new_roll = new_roll

  @classmethod
  def from_config(cls, cfg, training=False):
    search_area_factor = cfg.SEARCH.FACTOR
    output_sz = cfg.SEARCH.SIZE
    crop_type = cfg.CROP_TYPE
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
        "crop_type": crop_type,
        "transform": transform,
    }

  def _get_jittered_box(self, box, mode):
    """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

    if self.scale_jitter_factor.get('mode', 'gauss') == 'gauss':
      jittered_size = box[2:4] * torch.exp(
          torch.randn(2) * self.scale_jitter_factor[mode])
    elif self.scale_jitter_factor.get('mode', 'gauss') == 'uniform':
      jittered_size = box[2:4] * torch.exp(
          torch.FloatTensor(2).uniform_(-self.scale_jitter_factor[mode],
                                        self.scale_jitter_factor[mode]))
    else:
      raise Exception

    max_offset = (jittered_size.prod().sqrt() *
                  torch.tensor(self.center_jitter_factor[mode])).float()
    jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) -
                                                                0.5)

    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size),
                     dim=0)

  def __call__(self, data: TensorDict):
    # Apply joint transformations. i.e. All train/test frames in a sequence are applied the transformation with the
    # same parameters
    if self.transform['joint'] is not None:
      data['train_images'], data['train_anno'], data[
          'train_masks'] = self.transform['joint'](image=data['train_images'],
                                                   bbox=data['train_anno'],
                                                   mask=data['train_masks'])
      data['test_images'], data['test_anno'], data[
          'test_masks'] = self.transform['joint'](image=data['test_images'],
                                                  bbox=data['test_anno'],
                                                  mask=data['test_masks'],
                                                  new_roll=self.new_roll)

    for s in ['train', 'test']:
      assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
          "In pair mode, num train/test frames must be 1"

      # Add a uniform noise to the center pos
      jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
      orig_anno = data[s + '_anno']

      # Extract a crop containing the target
      crops, boxes, att_crops, mask_crops = prutils.target_image_crop(
          data[s + '_images'],
          jittered_anno,
          data[s + '_anno'],
          self.search_area_factor,
          self.output_sz,
          mode=self.crop_type,
          max_scale_change=self.max_scale_change,
          masks=data[s + '_masks'])

      # Apply independent transformations to each image
      data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[
          s + '_masks'] = self.transform[s](image=crops,
                                            bbox=boxes,
                                            att=att_crops,
                                            mask=mask_crops,
                                            joint=False)

    # Prepare output
    if self.mode == 'sequence':
      data = data.apply(stack_tensors)
    else:
      data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

    if self.box_mode == 'xyxy':
      data['search_boxes'] = box_convert(data['search_boxes'], 'xywh', 'xyxy')
      data['template_boxes'] = box_convert(data['template_boxes'], 'xywh',
                                           'xyxy')

    return data


@DATA_PROCESSING_REGISTRY.register()
class VOSSequenceProcessing(SiameseBaseProcessing):
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
    self.box_mode = box_mode
    self.max_scale_change = max_scale_change

  @classmethod
  def from_config(cls, cfg, training=False):
    search_area_factor = cfg.SEARCH.FACTOR
    output_sz = cfg.SEARCH.SIZE
    crop_type = cfg.CROP_TYPE
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
        "crop_type": crop_type,
        "transform": transform,
    }

  def _get_jittered_box(self, box):
    """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

    jittered_size = box[2:4] * \
        torch.exp(torch.randn(2) * self.scale_jitter_factor)
    max_offset = (jittered_size.prod().sqrt() *
                  torch.tensor(self.center_jitter_factor).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + \
        max_offset * (torch.rand(2) - 0.5)

    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size),
                     dim=0)

  def __call__(self, data: TensorDict):
    """
        args:
            data - The input data, should contain the following fields:
                 search_images', 'search_boxes'
        returns:
            TensorDict - output data block with following fields:
                'search_boxes', 'search_proposals', 'proposal_iou',
                'search_label' (optional), 'template_label' (optional), 'search_label_density' (optional), 'template_label_density' (optional)
        """

    # Add a uniform noise to the center pos
    jittered_boxes = [self._get_jittered_box(a) for a in data['search_boxes']]
    # self.visdom.register((crops[0], boxes[0]), 'Tracking', 1, 'Tracking')
    crops, boxes, att_mask, mask_crops = prutils.target_image_crop(
        data['search_images'],
        jittered_boxes,
        data['search_boxes'],
        self.search_area_factor,
        self.output_sz,
        mode=self.crop_type,
        masks=data['search_masks'])

    data['search_images'], data['search_boxes'], data['search_att'], data[
        'search_masks'] = self.transform['search'](image=crops,
                                                   bbox=boxes,
                                                   att=att_mask,
                                                   mask=mask_crops,
                                                   joint=False)

    # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
    # attention_mask = torch.stack(data['search_att'])
    # feat_size = self.output_sz // 16  # backbone stride for sot head
    # mask = F.interpolate(attention_mask[None].float(),
    #                      size=feat_size).to(torch.bool)
    # if mask.flatten(-2).all(-1).any():
    #   raise ValueError('mask wrong, will get nan loss nan')
    # self.visdom.register(
    #     ((data['search_images'][0].permute(1, 2, 0) *
    #       255).numpy().astype('uint8'), data['search_boxes'][0]), 'Tracking', 1,
    #     'Template')
    # self.visdom.register(
    #     ((data['search_images'][1].permute(1, 2, 0) *
    #       255).numpy().astype('uint8'), data['search_boxes'][1]), 'Tracking', 1,
    #     'Search')

    # Prepare output
    data = data.apply(stack_tensors)
    if self.box_mode == 'xyxy':
      data['search_boxes'] = box_convert(data['search_boxes'], 'xywh', 'xyxy')

    return data
