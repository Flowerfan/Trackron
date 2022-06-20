import torch
import torch.nn.functional as F
import math
import time
import cv2
from einops import rearrange

import trackron.data.bounding_box_utils as bbutils
import trackron.models.box_heads.box_regression as boxtrans
from trackron.structures import NestedTensor

from trackron.structures import TensorDict, TensorList
from trackron.utils.plotting import show_tensor, plot_graph
from trackron.libs import dcf
from trackron.models.layers import activation

from .base_tracker import BaseTracker
from .siamese_tracker import SiameseTracker
from .utils.preprocessing import sample_patch_multiscale, sample_patch_transformed, numpy_to_torch, sample_target
from .utils import augmentation
from .build import TRACKER_REGISTRY
from skimage import exposure
import numpy as np


@TRACKER_REGISTRY.register()
class StarkTracker(SiameseTracker):

  def setup(self):
    self.template_size = self.cfg.SOT.DATASET.TEMPLATE.SIZE
    self.template_factor = self.cfg.SOT.DATASET.TEMPLATE.FACTOR
    self.image_sample_size = self.cfg.SOT.DATASET.SEARCH.SIZE
    self.search_area_scale = self.cfg.SOT.DATASET.SEARCH.FACTOR
    self.output_score = self.cfg.TRACKER.OUTPUT_SCORE

  def init_sot(self, image, info: dict) -> dict:
    # Initialize some stuff
    self.frame_num = 1

    # Time initialization
    tic = time.time()

    # Get target position and size
    state = info['init_bbox']
    self.target_box = state
    self.pos = torch.Tensor(
        [state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
    self.target_sz = torch.Tensor([state[3], state[2]])
    self.image_sz = image.shape[:2]

    # Crop Image
    template_arr, scale_factor, template_mask_arr = sample_target(
        image,
        info['init_bbox'],
        self.template_factor,
        output_sz=self.template_size)
    # template_tensor = numpy_to_torch(template_arr).to(self.device)
    template_tensor = numpy_to_torch(template_arr).to(self.device) / 255.0
    template_box = self.get_cropped_img_box(self.pos, self.target_sz,
                                            self.pos, scale_factor).reshape(
                                                1, 4)  ## xyxy format
    #   mask_tensor = torch.from_numpy(template_mask_arr).to(torch.bool).to(
    #       self.device).view(1, 1, 128, 128)
    mask_tensor = torch.from_numpy(template_mask_arr).view(1, 128,
                                                           128).to(self.device)
    # self.visualize_search(numpy_to_torch(template_arr)[0])

    # Extract backbone feat
    with torch.no_grad():
      self.sot_ref_info = self.net.track_sot(template_tensor,
                                             mask_tensor,
                                             init_box=template_box)
    # save states
    self.target_box = info['init_bbox']

    out = {'time': time.time() - tic}
    out['target_bbox'] = self.target_box
    if self.output_score:
      out['target_bbox'] = self.target_box + [1.0]
    return out

  def add_image_mask(self, img, tl, br):
    tl = F.interpolate(tl.flatten().softmax(-1).view(1, 1, 20, 20),
                       (320, 320)).cpu().numpy()
    br = F.interpolate(br.flatten().softmax(-1).view(1, 1, 20, 20),
                       (320, 320)).cpu().numpy()
    # mask = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).to(
    #     torch.uint8).cpu().numpy()
    mask = tl + br
    mask = exposure.rescale_intensity(mask.squeeze(),
                                      out_range=(0, 255)).astype(np.uint8)
    mask = cv2.applyColorMap(mask.squeeze(), cv2.COLORMAP_JET)
    heat_img = cv2.addWeighted(mask, 0.5, img, 0.5, 0)
    return heat_img


  def visualize_search(self, image, box=None):
    if box is not None:
      box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
      self.visdom.register((image, box), 'Tracking', 1, 'SearchArea')
    else:
      self.visdom.register(image, 'image', 1, 'init')
