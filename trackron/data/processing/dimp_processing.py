import torch
import trackron.data.utils as prutils
from trackron.config import configurable
# from evaluation.utils.visdom import Visdom
from trackron.structures import TensorDict

from ..transforms import transforms as tfm
from .base import SiameseBaseProcessing, stack_tensors
from .build import DATA_PROCESSING_REGISTRY


@DATA_PROCESSING_REGISTRY.register()
class DiMPProcessing(SiameseBaseProcessing):
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
               proposal_params=None,
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
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
    super().__init__(*args, **kwargs)
    self.search_area_factor = search_area_factor
    self.output_sz = output_sz
    self.center_jitter_factor = center_jitter_factor
    self.scale_jitter_factor = scale_jitter_factor
    self.crop_type = crop_type
    self.mode = mode
    self.max_scale_change = max_scale_change

    self.proposal_params = proposal_params
    self.label_function_params = label_function_params
    # self.visdom = Visdom(1, None, {})

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
    proposal_params = {
        'min_iou': cfg.TRAIN.PROPOSALS.MIN_IOU,
        'boxes_per_target': cfg.TRAIN.PROPOSALS.BOXES_PER_TARGET,
        'sigma_factor': cfg.TRAIN.PROPOSALS.SIGMA_FACTOR
    }
    label_params = {
        'feature_sz': output_sz // cfg.FEATURE_STRIDE,
        'sigma_factor': cfg.LABLE_SIGMA,
        'kernel_sz': cfg.FILTER_SIZE
    }

    if training:
      transform = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                tfm.RandomHorizontalFlip(0.5))
      transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                      tfm.RandomHorizontalFlip(0.5))
    else:
      transform = tfm.Transform(tfm.ToTensorAndJitter(0.2))
      transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    return {
        "search_area_factor": search_area_factor,
        "output_sz": output_sz,
        "center_jitter_factor": center_jitter_factor,
        "scale_jitter_factor": scale_jitter_factor,
        "mode": mode,
        "transform": transform,
        "joint_transform": transform_joint,
        "proposal_params": proposal_params,
        "label_function_params": label_params
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

  def _generate_proposals(self, box):
    """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
    # Generate proposals
    num_proposals = self.proposal_params['boxes_per_target']
    proposal_method = self.proposal_params.get('proposal_method', 'default')

    if proposal_method == 'default':
      proposals = torch.zeros((num_proposals, 4))
      gt_iou = torch.zeros(num_proposals)

      for i in range(num_proposals):
        proposals[i, :], gt_iou[i] = prutils.perturb_box(
            box,
            min_iou=self.proposal_params['min_iou'],
            sigma_factor=self.proposal_params['sigma_factor'])
    elif proposal_method == 'gmm':
      proposals, _, _ = prutils.sample_box_gmm(
          box,
          self.proposal_params['proposal_sigma'],
          num_samples=num_proposals)
      gt_iou = prutils.iou(box.view(1, 4), proposals.view(-1, 4))
    else:
      raise ValueError('Unknown proposal method.')

    # Map to [-1, 1]
    gt_iou = gt_iou * 2 - 1
    return proposals, gt_iou

  def _generate_label_function(self, target_bb):
    """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

    gauss_label = prutils.gaussian_label_function(
        target_bb.view(-1, 4),
        self.label_function_params['sigma_factor'],
        self.label_function_params['kernel_sz'],
        self.label_function_params['feature_sz'],
        self.output_sz,
        end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

    return gauss_label

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
      assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
          "In pair mode, num template/search frames must be 1"
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
          self.output_sz,
          mode=self.crop_type,
          max_scale_change=self.max_scale_change)
      # self.visdom.register((crops[0], boxes[0]), 'Tracking', 1, 'Tracking')

      data[s + '_images'], data[s + '_boxes'] = self.transform[s](image=crops,
                                                                  bbox=boxes,
                                                                  joint=False)

    # Generate proposals
    if self.proposal_params:
      frame2_proposals, gt_iou = zip(
          *[self._generate_proposals(a) for a in data['search_boxes']])

      data['search_proposals'] = list(frame2_proposals)
      data['proposal_iou'] = list(gt_iou)

    # Prepare output
    if self.mode == 'sequence':
      data = data.apply(stack_tensors)
    else:
      data = data.apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)

    # Generate label functions
    if self.label_function_params is not None:
      data['template_label'] = self._generate_label_function(
          data['template_boxes'])
      data['search_label'] = self._generate_label_function(data['search_boxes'])

    return data


@DATA_PROCESSING_REGISTRY.register()
class KLDiMPProcessing(SiameseBaseProcessing):
  """ The processing class used for training PrDiMP that additionally supports the probabilistic classifier and
    bounding box regressor. See DiMPProcessing for details.
    """

  @configurable
  def __init__(self,
               search_area_factor,
               output_sz,
               center_jitter_factor,
               scale_jitter_factor,
               crop_type='inside_major',
               max_scale_change=None,
               mode='pair',
               proposal_params=None,
               label_function_params=None,
               label_density_params=None,
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
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
            label_density_params - Arguments for the label density generation process. See _generate_label_function for details.
        """
    super().__init__(*args, **kwargs)
    self.search_area_factor = search_area_factor
    self.output_sz = output_sz
    self.center_jitter_factor = center_jitter_factor
    self.scale_jitter_factor = scale_jitter_factor
    self.crop_type = crop_type
    self.mode = mode
    self.max_scale_change = max_scale_change

    self.proposal_params = proposal_params
    self.label_function_params = label_function_params
    self.label_density_params = label_density_params

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
    label_params = {
        'feature_sz': output_sz // cfg.FEATURE_STRIDE,
        'sigma_factor': cfg.LABLE_SIGMA,
        'kernel_sz': cfg.FILTER_SIZE
    }
    label_density_params = {
        'feature_sz': output_sz // cfg.FEATURE_STRIDE,
        'sigma_factor': cfg.LABLE_SIGMA,
        'kernel_sz': cfg.FILTER_SIZE
    }
    proposal_params = {
        "boxes_per_frame": 128,
        "gt_sigma": (0.05, 0.05),
        "proposal_sigma": [(0.05, 0.05), (0.5, 0.5)]
    }

    if training:
      transform = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                tfm.RandomHorizontalFlip(0.5))
      transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                      tfm.RandomHorizontalFlip(0.5))
    else:
      transform = tfm.Transform(tfm.ToTensorAndJitter(0.2))
      transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    return {
        "search_area_factor": search_area_factor,
        "output_sz": output_sz,
        "center_jitter_factor": center_jitter_factor,
        "scale_jitter_factor": scale_jitter_factor,
        "mode": mode,
        "crop_type": "inside_major",
        "max_scale_change": 1.5,
        "proposal_params": proposal_params,
        "label_function_params": label_params,
        "label_density_params": label_density_params,
        "transform": transform,
        "joint_transform": transform_joint
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

  def _generate_proposals(self, box):
    """ Generate proposal sample boxes from a GMM proposal distribution and compute their ground-truth density.
        This is used for ML and KL based regression learning of the bounding box regressor.
        args:
            box - input bounding box
        """
    # Generate proposals
    proposals, proposal_density, gt_density = prutils.sample_box_gmm(
        box,
        self.proposal_params['proposal_sigma'],
        gt_sigma=self.proposal_params['gt_sigma'],
        num_samples=self.proposal_params['boxes_per_frame'],
        add_mean_box=self.proposal_params.get('add_mean_box', False))

    return proposals, proposal_density, gt_density

  def _generate_label_function(self, target_bb):
    """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

    gauss_label = prutils.gaussian_label_function(
        target_bb.view(-1, 4),
        self.label_function_params['sigma_factor'],
        self.label_function_params['kernel_sz'],
        self.label_function_params['feature_sz'],
        self.output_sz,
        end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

    return gauss_label

  def _generate_template_label_function(self, target_bb):
    """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
    gauss_label = prutils.gaussian_label_function(
        target_bb.view(-1, 4),
        0.1,
        self.label_function_params['kernel_sz'],
        self.label_function_params['feature_sz'],
        self.output_sz,
        end_pad_if_even=False)
    return gauss_label

  def _generate_label_density(self, target_bb):
    """ Generates the gaussian label density centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

    feat_sz = self.label_density_params['feature_sz'] * \
        self.label_density_params.get('interp_factor', 1)
    gauss_label = prutils.gaussian_label_function(
        target_bb.view(-1, 4),
        self.label_density_params['sigma_factor'],
        self.label_density_params['kernel_sz'],
        feat_sz,
        self.output_sz,
        end_pad_if_even=self.label_density_params.get('end_pad_if_even', True),
        density=True,
        uni_bias=self.label_density_params.get('uni_weight', 0.0))

    gauss_label *= (gauss_label > self.label_density_params.get(
        'threshold', 0.0)).float()

    if self.label_density_params.get('normalize', False):
      g_sum = gauss_label.sum(dim=(-2, -1))
      valid = g_sum > 0.01
      gauss_label[valid, :, :] /= g_sum[valid].view(-1, 1, 1)
      gauss_label[~valid, :, :] = 1.0 / \
          (gauss_label.shape[-2] * gauss_label.shape[-1])

    gauss_label *= 1.0 - self.label_density_params.get('shrink', 0.0)

    return gauss_label

  def __call__(self, data: TensorDict):
    """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_boxes', 'search_boxes'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_boxes', 'search_boxes', 'search_proposals', 'proposal_density', 'gt_density',
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
      assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
          "In pair mode, num template/search frames must be 1"
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
          self.output_sz,
          mode=self.crop_type,
          max_scale_change=self.max_scale_change)

      data[s + '_images'], data[s + '_boxes'] = self.transform[s](image=crops,
                                                                  bbox=boxes,
                                                                  joint=False)

    # Generate proposals
    proposals, proposal_density, gt_density = zip(
        *[self._generate_proposals(a) for a in data['search_boxes']])

    data['search_proposals'] = proposals
    data['proposal_density'] = proposal_density
    data['gt_density'] = gt_density

    for s in ['template', 'search']:
      is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
      if is_distractor is not None:
        for is_dist, box in zip(is_distractor, data[s + '_boxes']):
          if is_dist:
            box[0] = 99999999.9
            box[1] = 99999999.9

    # Prepare output
    if self.mode == 'sequence':
      data = data.apply(stack_tensors)
    else:
      data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

    # Generate label functions
    if self.label_function_params is not None:
      # data['template_label'] = self._generate_label_function(
      #     data['template_boxes'])
      data['template_label'] = self._generate_template_label_function(
          data['template_boxes'])
      data['search_label'] = self._generate_label_function(data['search_boxes'])
    if self.label_density_params is not None:
      data['template_label_density'] = self._generate_label_density(
          data['template_boxes'])
      data['search_label_density'] = self._generate_label_density(
          data['search_boxes'])

    return data
