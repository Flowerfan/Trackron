import logging
import jpeg4py
import cv2 as cv
from PIL import Image
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3,
                          1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128],
                         [128, 128, 128], [64, 0, 0], [191, 0, 0], [64, 128, 0],
                         [191, 128, 0], [64, 0, 128], [191, 0, 128],
                         [64, 128, 128], [191, 128, 128], [0, 64, 0],
                         [128, 64, 0], [0, 191, 0], [128, 191, 0], [0, 64, 128],
                         [128, 64, 128]]


def default_image_loader(path):
  """The default image loader, reads the image from the given path. It first tries to use the jpeg4py_loader,
    but reverts to the opencv_loader if the former is not available."""
  if default_image_loader.use_jpeg4py is None:
    # Try using jpeg4py
    im = jpeg4py_loader(path)
    if im is None:
      default_image_loader.use_jpeg4py = False
      logger = logging.getLogger(__name__)
      logger.info('Using opencv_loader instead.')
    else:
      default_image_loader.use_jpeg4py = True
      return im
  if default_image_loader.use_jpeg4py:
    return jpeg4py_loader(path)
  return opencv_loader(path)


default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
  """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
  try:
    return jpeg4py.JPEG(path).decode()
  except Exception as e:
    logger = logging.getLogger(__name__)
    logger.info('ERROR: Could not read image "{}"'.format(path))
    logger.info(e)
    return None


def opencv_loader(path):
  """ Read image using opencv's imread function and returns it in rgb format"""
  try:
    im = cv.imread(path, cv.IMREAD_COLOR)

    # convert to rgb and return
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)
  except Exception as e:
    print('ERROR: Could not read image "{}"'.format(path))
    print(e)
    return None


def jpeg4py_loader_w_failsafe(path):
  """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
  try:
    return jpeg4py.JPEG(path).decode()
  except:
    try:
      im = cv.imread(path, cv.IMREAD_COLOR)

      # convert to rgb and return
      return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
      print('ERROR: Could not read image "{}"'.format(path))
      print(e)
      return None


def opencv_seg_loader(path):
  """ Read segmentation annotation using opencv's imread function"""
  try:
    return cv.imread(path)
  except Exception as e:
    print('ERROR: Could not read image "{}"'.format(path))
    print(e)
    return None


def imread_indexed(filename):
  """ Load indexed image with given filename. Used to read segmentation annotations."""

  im = Image.open(filename)

  annotation = np.atleast_3d(im)[..., 0]
  return annotation


def imwrite_indexed(filename, array, color_palette=None):
  """ Save indexed image as png. Used to save segmentation annotation."""

  if color_palette is None:
    color_palette = davis_palette

  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im.putpalette(color_palette.ravel())
  im.save(filename, format='PNG')


def clone_modules(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
  """Return an activation function given a string"""
  if activation == "relu":
    return F.relu
  if activation == "gelu":
    return F.gelu
  if activation == "glu":
    return F.glu
  raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def inverse_sigmoid(x, eps=1e-5):
  x = x.clamp(min=0, max=1)
  x1 = x.clamp(min=eps)
  x2 = (1 - x).clamp(min=eps)
  return torch.log(x1 / x2)

def masks_to_boxes(masks):
  """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
  if masks.numel() == 0:
    return torch.zeros((0, 4), device=masks.device)

  h, w = masks.shape[-2:]

  y = torch.arange(0, h, dtype=torch.float)
  x = torch.arange(0, w, dtype=torch.float)
  y, x = torch.meshgrid(y, x)

  x_mask = (masks * x.unsqueeze(0))
  x_max = x_mask.flatten(1).max(-1)[0]
  x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

  y_mask = (masks * y.unsqueeze(0))
  y_max = y_mask.flatten(1).max(-1)[0]
  y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

  return torch.stack([x_min, y_min, x_max, y_max], 1)
