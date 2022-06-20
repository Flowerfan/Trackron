import torch
import math
import cv2 as cv
import random
import numpy as np
import torch.nn.functional as F
from .bounding_box_utils import rect_to_rel, rel_to_rect
import torchvision.transforms.functional as vf
from torchvision.ops import box_convert


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
  """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
  if not isinstance(target_bb, list):
    x, y, w, h = target_bb.tolist()
  else:
    x, y, w, h = target_bb
  # Crop image
  crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

  if crop_sz < 1:
    raise Exception('Too small bounding box.')

  x1 = round(x + 0.5 * w - crop_sz * 0.5)
  x2 = x1 + crop_sz

  y1 = round(y + 0.5 * h - crop_sz * 0.5)
  y2 = y1 + crop_sz

  x1_pad = max(0, -x1)
  x2_pad = max(x2 - im.shape[1] + 1, 0)

  y1_pad = max(0, -y1)
  y2_pad = max(y2 - im.shape[0] + 1, 0)

  # Crop target
  im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
  if mask is not None:
    mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

  # Pad
  im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                     cv.BORDER_CONSTANT)
  # deal with attention mask
  H, W, _ = im_crop_padded.shape
  att_mask = np.ones((H, W))
  end_x, end_y = -x2_pad, -y2_pad
  if y2_pad == 0:
    end_y = None
  if x2_pad == 0:
    end_x = None
  att_mask[y1_pad:end_y, x1_pad:end_x] = 0
  if mask is not None:
    mask_crop_padded = F.pad(mask_crop,
                             pad=(x1_pad, x2_pad, y1_pad, y2_pad),
                             mode='constant',
                             value=0)

  if output_sz is not None:
    resize_factor = output_sz / crop_sz
    im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
    att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
    if mask is None:
      return im_crop_padded, resize_factor, att_mask
    mask_crop_padded = \
    F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
    return im_crop_padded, resize_factor, att_mask, mask_crop_padded

  else:
    if mask is None:
      return im_crop_padded, att_mask.astype(np.bool_), 1.0
    return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def jittered_center_crop(frames,
                         box_extract,
                         box_gt,
                         search_area_factor,
                         output_sz,
                         masks=None):
  """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

  if masks is None:
    crops_resize_factors = [
        sample_target(f, a, search_area_factor, output_sz)
        for f, a in zip(frames, box_extract)
    ]
    frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
    masks_crop = None
  else:
    crops_resize_factors = [
        sample_target(f, a, search_area_factor, output_sz, m)
        for f, a, m in zip(frames, box_extract, masks)
    ]
    frames_crop, resize_factors, att_mask, masks_crop = zip(
        *crops_resize_factors)
  # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
  crop_sz = torch.Tensor([output_sz, output_sz])

  # find the bb location in the crop
  '''Note that here we use normalized coord'''
  box_crop = [
      transform_image_to_crop(a_gt, a_ex, rf, crop_sz)
      for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)
  ]  # (x1,y1,w,h) list of tensors

  return frames_crop, box_crop, att_mask, masks_crop


def transform_image_to_crop(box_in: torch.Tensor,
                            box_extract: torch.Tensor,
                            resize_factor: float,
                            crop_sz: torch.Tensor,
                            normalize=False) -> torch.Tensor:
  """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
  box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

  box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

  box_out_center = (crop_sz - 1) / 2 + (box_in_center -
                                        box_extract_center) * resize_factor
  box_out_wh = box_in[2:4] * resize_factor

  box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
  if normalize:
    return box_out / crop_sz[0]
  else:
    return box_out


def sample_target_adaptive(im,
                           target_bb,
                           search_area_factor,
                           output_sz,
                           mode: str = 'replicate',
                           max_scale_change=None,
                           mask=None):
  """ Extracts a crop centered at target_bb box, of area search_area_factor^2. If the crop area contains regions
    outside the image, it is shifted so that the it is inside the image. Further, if the crop area exceeds the image
    size, a smaller crop which fits the image is returned instead.

    args:
        im - Input numpy image to crop.
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done. [W, H]
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
        max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
        mask - Optional mask to apply the same crop.

    returns:
        numpy image - Extracted crop.
        torch.Tensor - A bounding box denoting the cropped region in the image.
        numpy mask - Cropped mask returned only if mask is not None.
    """

  if max_scale_change is None:
    max_scale_change = float('inf')
  if isinstance(output_sz, (float, int)):
    output_sz = (output_sz, output_sz)
  output_sz = torch.Tensor(output_sz)

  im_h = im.shape[0]
  im_w = im.shape[1]

  bbx, bby, bbw, bbh = target_bb.tolist()

  # Crop image
  crop_sz_x, crop_sz_y = (output_sz *
                          (target_bb[2:].prod() / output_sz.prod()).sqrt() *
                          search_area_factor).ceil().long().tolist()

  # Get new sample size if forced inside the image
  if mode == 'inside' or mode == 'inside_major':
    # Calculate rescaling factor if outside the image
    rescale_factor = [crop_sz_x / im_w, crop_sz_y / im_h]
    if mode == 'inside':
      rescale_factor = max(rescale_factor)
    elif mode == 'inside_major':
      rescale_factor = min(rescale_factor)
    rescale_factor = min(max(1, rescale_factor), max_scale_change)

    crop_sz_x = math.floor(crop_sz_x / rescale_factor)
    crop_sz_y = math.floor(crop_sz_y / rescale_factor)

  if crop_sz_x < 1 or crop_sz_y < 1:
    raise Exception('Too small bounding box.')

  x1 = round(bbx + 0.5 * bbw - crop_sz_x * 0.5)
  x2 = x1 + crop_sz_x

  y1 = round(bby + 0.5 * bbh - crop_sz_y * 0.5)
  y2 = y1 + crop_sz_y

  # Move box inside image
  shift_x = max(0, -x1) + min(0, im_w - x2)
  x1 += shift_x
  x2 += shift_x

  shift_y = max(0, -y1) + min(0, im_h - y2)
  y1 += shift_y
  y2 += shift_y

  out_x = (max(0, -x1) + max(0, x2 - im_w)) // 2
  out_y = (max(0, -y1) + max(0, y2 - im_h)) // 2
  shift_x = (-x1 - out_x) * (out_x > 0)
  shift_y = (-y1 - out_y) * (out_y > 0)

  x1 += shift_x
  x2 += shift_x
  y1 += shift_y
  y2 += shift_y

  x1_pad = max(0, -x1)
  x2_pad = max(x2 - im.shape[1] + 1, 0)

  y1_pad = max(0, -y1)
  y2_pad = max(y2 - im.shape[0] + 1, 0)

  # Crop target
  im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

  if mask is not None:
    mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

  # Pad
  im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                     cv.BORDER_REPLICATE)
  ### attention mask
  H, W, _ = im_crop_padded.shape
  att_mask = np.ones((H, W))
  end_x, end_y = -x2_pad, -y2_pad
  if y2_pad == 0:
    end_y = None
  if x2_pad == 0:
    end_x = None
  att_mask[y1_pad:end_y, x1_pad:end_x] = 0

  if mask is not None:
    mask_crop_padded = F.pad(mask_crop,
                             pad=(x1_pad, x2_pad, y1_pad, y2_pad),
                             mode='constant',
                             value=0)

  # Resize image
  im_out = cv.resize(im_crop_padded, tuple(output_sz.long().tolist()))
  att_mask = cv.resize(att_mask, tuple(output_sz.long().tolist()))

  if mask is not None:
    mask_out = \
    F.interpolate(mask_crop_padded[None, None], tuple(output_sz.flip(0).long().tolist()), mode='nearest')[0, 0]

  crop_box = torch.Tensor([x1, y1, x2 - x1, y2 - y1])

  if mask is None:
    return im_out, crop_box, att_mask
  return im_out, crop_box, att_mask, mask_out


def crop_and_resize(im, box, crop_bb, output_sz, mask=None):
  if isinstance(output_sz, (float, int)):
    output_sz = (output_sz, output_sz)

  im_h = im.shape[0]
  im_w = im.shape[1]

  if crop_bb[2] < 1 or crop_bb[3] < 1:
    raise Exception('Too small bounding box.')

  x1 = crop_bb[0]
  x2 = crop_bb[0] + crop_bb[2]

  y1 = crop_bb[1]
  y2 = crop_bb[1] + crop_bb[3]

  x1_pad = max(0, -x1)
  x2_pad = max(x2 - im.shape[1] + 1, 0)

  y1_pad = max(0, -y1)
  y2_pad = max(y2 - im.shape[0] + 1, 0)

  # Crop target
  im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

  if mask is not None:
    mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

  # Pad
  im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                     cv.BORDER_REPLICATE)

  if mask is not None:
    mask_crop_padded = F.pad(mask_crop,
                             pad=(x1_pad, x2_pad, y1_pad, y2_pad),
                             mode='constant',
                             value=0)

  # Resize image
  im_out = cv.resize(im_crop_padded, output_sz)

  if mask is not None:
    mask_out = F.interpolate(mask_crop_padded[None, None],
                             (output_sz[1], output_sz[0]),
                             mode='nearest')[0, 0]

  rescale_factor = output_sz[0] / crop_bb[2]

  # Hack
  if box is not None:
    box_crop = box.clone()
    box_crop[0] -= crop_bb[0]
    box_crop[1] -= crop_bb[1]

    box_crop *= rescale_factor
  else:
    box_crop = None

  if mask is None:
    return im_out, box_crop
  else:
    return im_out, box_crop, mask_out


def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor,
                          crop_sz: torch.Tensor) -> torch.Tensor:
  """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - [x,y,w,h] the box for which the co-ordinates are to be transformed
        crop_box - [x, y, w, h] bounding box defining the crop in the original image
        crop_sz - [w, h] size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

  box_out = box.clone()
  box_out[:2] -= crop_box[:2]

  scale_factor = crop_sz / crop_box[2:]

  box_out[:2] *= scale_factor
  box_out[2:] *= scale_factor
  return box_out


def target_image_crop(frames,
                      box_extract,
                      box_gt,
                      search_area_factor,
                      output_sz,
                      mode: str = 'replicate',
                      max_scale_change=None,
                      masks=None):
  """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. If the crop area contains regions outside the image, it is shifted / shrunk so that it
    completely fits inside the image. The extracted crops are then resized to output_sz. Further, the co-ordinates of
    the box box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized [h, w]
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
        max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
        masks - Optional masks to apply the same crop.

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

  if isinstance(output_sz, (float, int)):
    output_sz = (output_sz, output_sz)
  else:
    output_sz = output_sz[::-1]   ## [h w] to [w h]

  if masks is None:
    frame_crops_boxes = [
        sample_target_adaptive(f, a, search_area_factor, output_sz, mode,
                               max_scale_change)
        for f, a in zip(frames, box_extract)
    ]

    frames_crop, crop_boxes, att_crop = zip(*frame_crops_boxes)
  else:
    frame_crops_boxes_masks = [
        sample_target_adaptive(f,
                               a,
                               search_area_factor,
                               output_sz,
                               mode,
                               max_scale_change,
                               mask=m)
        for f, a, m in zip(frames, box_extract, masks)
    ]

    frames_crop, crop_boxes, att_crop, masks_crop = zip(
        *frame_crops_boxes_masks)

  crop_sz = torch.Tensor(output_sz)

  # find the bb location in the crop
  box_crop = [
      transform_box_to_crop(bb_gt, crop_bb, crop_sz)
      for bb_gt, crop_bb in zip(box_gt, crop_boxes)
  ]

  if masks is None:
    return frames_crop, box_crop, att_crop
  else:
    return frames_crop, box_crop, att_crop, masks_crop


def scale_shortedge_image(im, bbox_gt, short_edge_length, max_size: int = 1333):
  h, w = im.shape[:2]
  size = np.random.choice(short_edge_length)

  scale = size * 1.0 / min(h, w)
  if h < w:
    newh, neww = size, scale * w
  else:
    newh, neww = scale * h, size
  if max(newh, neww) > max_size:
    scale = max_size * 1.0 / max(newh, neww)
    newh = newh * scale
    neww = neww * scale
  neww = int(neww + 0.5)
  newh = int(newh + 0.5)
  image, box = scale_fixsize_image(im, bbox_gt, (newh, neww))
  return image, box


def scale_fixshortedge_image(im, bbox_gt, short_edge_length):
  h, w = im.shape[:2]
  size = np.random.choice(short_edge_length)
  image, box = scale_fixsize_image(im, bbox_gt, (size, size))
  return image, box


def crop_fixsize_image(im,
                       target_bb,
                       bbox_gt,
                       output_sz,
                       mode: str = 'replicate'):
  """ Extracts a crop centered at target_bb box, of area search_area_factor^2. If the crop area contains regions
    outside the image, it is shifted so that the it is inside the image. Further, if the crop area exceeds the image
    size, a smaller crop which fits the image is returned instead.

    args:
        im - Input numpy image to crop.
        target_bb - target box [x, y, w, h]
        box_gt - The co-ordinates of target box is transformed from image co-ordinates to the crop co-ordinates
        output_sz - [h,w] (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
    returns:
        numpy image - Extracted crop.
        torch.Tensor - A bounding box denoting the cropped region in the image.
    """

  if isinstance(output_sz, (float, int)):
    output_sz = (output_sz, output_sz)

  im_h = im.shape[0]
  im_w = im.shape[1]

  bbx, bby, bbw, bbh = target_bb.tolist()

  # Crop image
  crop_sz_x, crop_sz_y = output_sz[1], output_sz[0]

  x1 = round(bbx + 0.5 * bbw - crop_sz_x * 0.5)
  x2 = x1 + crop_sz_x

  y1 = round(bby + 0.5 * bbh - crop_sz_y * 0.5)
  y2 = y1 + crop_sz_y

  # Move box inside image
  shift_x = max(0, -x1) + min(0, im_w - x2)
  x1 += shift_x
  x2 += shift_x

  shift_y = max(0, -y1) + min(0, im_h - y2)
  y1 += shift_y
  y2 += shift_y

  out_x = (max(0, -x1) + max(0, x2 - im_w)) // 2
  out_y = (max(0, -y1) + max(0, y2 - im_h)) // 2
  shift_x = (-x1 - out_x) * (out_x > 0)
  shift_y = (-y1 - out_y) * (out_y > 0)

  x1 += shift_x
  x2 += shift_x
  y1 += shift_y
  y2 += shift_y

  x1_pad = max(0, -x1)
  x2_pad = max(x2 - im.shape[1] + 1, 0)

  y1_pad = max(0, -y1)
  y2_pad = max(y2 - im.shape[0] + 1, 0)

  # Crop target
  im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

  # Pad
  im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                     cv.BORDER_REPLICATE)

  crop_box = torch.Tensor([x1, y1, x2 - x1, y2 - y1])

  crop_sz = torch.Tensor([crop_sz_x, crop_sz_y])

  # find the bb location in the crop
  box_crop = transform_box_to_crop(bbox_gt, crop_box, crop_sz)

  return im_crop_padded, box_crop


def scale_fixsize_image(im,
                        bbox_gt,
                        output_sz,
                        mode: str = 'replicate',
                        min_size=32,
                        max_size=96):
  """ Extracts a crop centered at target_bb box, of area search_area_factor^2. If the crop area contains regions
    outside the image, it is shifted so that the it is inside the image. Further, if the crop area exceeds the image
    size, a smaller crop which fits the image is returned instead.

    args:
        im - Input numpy image to crop.
        target_bb - target box [x, y, w, h]
        box_gt - The co-ordinates of target box is transformed from image co-ordinates to the crop co-ordinates
        output_sz - [h,w] (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.
        mode - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
               If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
               If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
    returns:
        numpy image - Extracted crop.
        torch.Tensor - A bounding box denoting the cropped region in the image.
    """

  def _get_jittered_box(box, scale_factor=1.0, shift_factor=3.0):
    """ Jitter the input box
        args:
            box - input bounding box

        returns:
            torch.Tensor - jittered box
        """
    # jittered_size = box[2:4] * \
    #     torch.exp(torch.randn(2) * scale_factor)

    jittered_size = box[2:4] * scale_factor
    max_offset = (jittered_size.prod().sqrt() *
                  torch.tensor(shift_factor).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + \
        max_offset * (torch.rand(2) - 0.5)

    return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size),
                     dim=0)

  if isinstance(output_sz, (float, int)):
    output_sz = (output_sz, output_sz)
  output_sz = torch.tensor(output_sz)

  im_h = im.shape[0]
  im_w = im.shape[1]

  # Crop image
  bb_area = bbox_gt[2:].prod().sqrt()
  min_scale = (bb_area / max_size)
  max_scale = (bb_area / min_size)
  s = np.random.rand()
  scale = s * min_scale + (1 - s) * max_scale
  crop_sz_y, crop_sz_x = (output_sz * scale).long().tolist()

  target_bb = _get_jittered_box(bbox_gt, scale)
  bbx, bby, bbw, bbh = target_bb.tolist()
  gbx1, gby1, gbw, gbh = bbox_gt.ceil().long().tolist()
  gbx2, gby2 = gbx1 + gbw, gby1 + gbh
  x1 = round(bbx + 0.5 * bbw - crop_sz_x * 0.5)
  x2 = x1 + crop_sz_x

  y1 = round(bby + 0.5 * bbh - crop_sz_y * 0.5)
  y2 = y1 + crop_sz_y

  ## make croped image include target object
  if gbx1 >= gbx2 or gby1 >= gby2 or gbw < 0 or gbh < 0:
    pass
    # print(bbox_gt)
    # import pdb; pdb.set_trace()
  shift_x = min(0, gbx1 - x1) + max(0, gbx2 - x2)
  x1 += shift_x
  x2 += shift_x
  shift_y = min(0, gby1 - y1) + max(0, gby2 - y2)
  y1 += shift_y
  y2 += shift_y

  # Move box inside image
  shift_x = max(0, -x1) + min(0, im_w - x2)
  x1 += shift_x
  x2 += shift_x

  shift_y = max(0, -y1) + min(0, im_h - y2)
  y1 += shift_y
  y2 += shift_y

  out_x = (max(0, -x1) + max(0, x2 - im_w)) // 2
  out_y = (max(0, -y1) + max(0, y2 - im_h)) // 2
  shift_x = (-x1 - out_x) * (out_x > 0)
  shift_y = (-y1 - out_y) * (out_y > 0)

  x1 += shift_x
  x2 += shift_x
  y1 += shift_y
  y2 += shift_y

  x1_pad = max(0, -x1)
  x2_pad = max(x2 - im.shape[1] + 1, 0)

  y1_pad = max(0, -y1)
  y2_pad = max(y2 - im.shape[0] + 1, 0)

  # Crop target
  im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

  # Pad
  im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad,
                                     cv.BORDER_REPLICATE)

  im_out = cv.resize(im_crop_padded, (output_sz[1], output_sz[0]))

  # rescale_factor = output_sz[0] / crop_bb[2]

  # Hack
  # if box is not None:
  #     box_crop = box.clone()
  #     box_crop[0] -= crop_bb[0]
  #     box_crop[1] -= crop_bb[1]

  #     box_crop *= rescale_factor
  # else:
  #     box_crop = None

  crop_box = torch.Tensor([x1, y1, x2 - x1, y2 - y1])

  crop_sz = torch.Tensor([crop_sz_x, crop_sz_y])

  x, y, w, h = bbox_gt

  new_x = max(x, x1)
  new_y = max(y, y1)
  new_w = min(x + w, x2) - new_x
  new_h = min(y + h, y2) - new_y
  # shift_x = min(0, x-crop_box[0])
  # shift_y = min(0, y-crop_box[1])
  # x = x + shift_x
  # y = y + shift_y
  # w = w + shift_x
  # h = h + shift_y
  # shift_w = min(0, crop_box[0]+crop_box[2]-x-w)
  # shift_y = min(0, crop_box[1]+crop_box[3]-y-h)
  # w = w + shift_w
  # h = h + shift_y
  bbox_gt = torch.tensor([new_x, new_y, new_w, new_h])
  # find the bb location in the crop
  box_crop = transform_box_to_crop(bbox_gt, crop_box, output_sz.flip(0))

  return im_out, box_crop

def resize_image(image, input_size):
  if len(image.shape) == 3:
    padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
  else:
    padded_img = np.ones(input_size) * 114.0
  img = np.array(image)
  r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
  resized_img = cv.resize(
      img,
      (int(img.shape[1] * r), int(img.shape[0] * r)),
      interpolation=cv.INTER_LINEAR,
  ).astype(np.float32)
  padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
  return padded_img


def resize_image_boxes(image, box, output_sz):
  """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. If the crop area contains regions outside the image, it is shifted / shrunk so that it
    completely fits inside the image. The extracted crops are then resized to output_sz. Further, the co-ordinates of
    the box box_gt are transformed to the image crop co-ordinates

    args:
        frames - image
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of resized image 
        list - box_gt location in the crop co-ordinates
        """

  if isinstance(output_sz, (float, int)):
    output_sz = (output_sz, output_sz)
  im_sz = image.shape
  w_scale = output_sz[1] / im_sz[1]
  h_scale = output_sz[0] / im_sz[0]
  image = cv.resize(image, (output_sz[1], output_sz[0]))
  box = torch.tensor(
      [box[0] * w_scale, box[1] * h_scale, box[2] * w_scale, box[3] * h_scale])
  return image, box

def iou_all(reference, proposals):
  """Compute the IoU between a reference box with multiple proposal boxes.

    args:
        reference - Tensor of shape 4.
        proposals - Tensor of shape (..., 4)

    returns:
        torch.Tensor - Tensor of shape (...,) containing IoU of reference box with each proposal box.
    """

  # Intersection box
  tl = torch.max(reference[:2], proposals[..., :2])
  br = torch.min(reference[:2] + reference[2:],
                 proposals[..., :2] + proposals[..., 2:])
  sz = (br - tl).clamp(0)

  # Area
  intersection = sz.prod(dim=-1)
  union = reference[2:].prod() + proposals[..., 2:].prod(
      dim=-1) - intersection

  return intersection / union

def iou(reference, proposals):
  """Compute the IoU between a reference box with multiple proposal boxes.

    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)

    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """

  # Intersection box
  tl = torch.max(reference[:, :2], proposals[:, :2])
  br = torch.min(reference[:, :2] + reference[:, 2:],
                 proposals[:, :2] + proposals[:, 2:])
  sz = (br - tl).clamp(0)

  # Area
  intersection = sz.prod(dim=1)
  union = reference[:, 2:].prod(dim=1) + proposals[:, 2:].prod(
      dim=1) - intersection

  return intersection / union


def rand_uniform(a, b, shape=1):
  """ sample numbers uniformly between a and b.
    args:
        a - lower bound
        b - upper bound
        shape - shape of the output tensor

    returns:
        torch.Tensor - tensor of shape=shape
    """
  return (b - a) * torch.rand(shape) + a


def perturb_box(box, min_iou=0.5, sigma_factor=0.1):
  """ Perturb the input box by adding gaussian noise to the co-ordinates

     args:
        box - input box
        min_iou - minimum IoU overlap between input box and the perturbed box
        sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                        sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                        sigma_factor element can be either a float, or a tensor
                        of shape (4,) specifying the sigma_factor per co-ordinate

    returns:
        torch.Tensor - the perturbed box
    """

  if isinstance(sigma_factor, list):
    # If list, sample one sigma_factor as current sigma factor
    c_sigma_factor = random.choice(sigma_factor)
  else:
    c_sigma_factor = sigma_factor

  if not isinstance(c_sigma_factor, torch.Tensor):
    c_sigma_factor = c_sigma_factor * torch.ones(4)

  perturb_factor = torch.sqrt(box[2] * box[3]) * c_sigma_factor.to(box.device)

  # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
  for i_ in range(100):
    c_x = box[0] + 0.5 * box[2]
    c_y = box[1] + 0.5 * box[3]
    c_x_per = random.gauss(c_x, perturb_factor[0])
    c_y_per = random.gauss(c_y, perturb_factor[1])

    w_per = random.gauss(box[2], perturb_factor[2])
    h_per = random.gauss(box[3], perturb_factor[3])

    if w_per <= 1:
      w_per = box[2] * rand_uniform(0.15, 0.5).to(box.device)

    if h_per <= 1:
      h_per = box[3] * rand_uniform(0.15, 0.5).to(box.device)

    box_per = torch.tensor(
        [c_x_per - 0.5 * w_per, c_y_per - 0.5 * h_per, w_per, h_per],
        device=box.device).round()

    if box_per[2] <= 1:
      box_per[2] = box[2] * rand_uniform(0.15, 0.5).to(box.device)

    if box_per[3] <= 1:
      box_per[3] = box[3] * rand_uniform(0.15, 0.5).to(box.device)

    box_iou = iou(box.view(1, 4), box_per.view(1, 4))

    # if there is sufficient overlap, return
    if box_iou > min_iou:
      return box_per, box_iou

    # else reduce the perturb factor
    perturb_factor *= 0.9

  return box_per, box_iou


def gauss_1d(sz, sigma, center, end_pad=0, density=False):
  k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
  gauss = torch.exp(-1.0 / (2 * sigma**2).view(-1, 1) *
                    (k - center.reshape(-1, 1))**2)
  if density:
    gauss /= math.sqrt(2 * math.pi) * sigma
  return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
  if isinstance(sigma, (float, int)):
    sigma = torch.tensor((sigma, sigma)).view(1, 2)
  return gauss_1d(sz[0].item(), sigma[:, 0], center[:, 0], end_pad[0], density).reshape(center.shape[0], -1, 1) * \
         gauss_1d(sz[1].item(), sigma[:, 1], center[:, 1], end_pad[1], density).reshape(center.shape[0], 1, -1)


def gaussian_center_label(target_center,
                          sigma_factor,
                          kernel_sz,
                          feat_sz,
                          image_sz,
                          end_pad_if_even=True,
                          density=False,
                          uni_bias=0):
  """Construct Gaussian label function.
    args: 
        target_center: shape [N, 2] in the format [h,w]
        kernel_sz, shape [2] indicates keneral[h,w]
        feat_sz, shape [2] indicates feature [h,w]
        image_sz, [h, w]
    return:
        label_map: [N,H,W]
    """

  if isinstance(kernel_sz, (float, int)):
    kernel_sz = (kernel_sz, kernel_sz)
  if isinstance(feat_sz, (float, int)):
    feat_sz = (feat_sz, feat_sz)
  if isinstance(image_sz, (float, int)):
    image_sz = (image_sz, image_sz)

  image_sz = torch.Tensor(image_sz)
  feat_sz = torch.Tensor(feat_sz)

  target_center_norm = (target_center - image_sz / 2) / image_sz

  center = feat_sz * target_center_norm + 0.5 * \
           torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

  sigma = sigma_factor * feat_sz.prod().sqrt().item()

  if end_pad_if_even:
    end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
  else:
    end_pad = (0, 0)
  gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
  if density:
    sz = (feat_sz + torch.Tensor(end_pad)).prod()
    label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
  else:
    label = gauss_label + uni_bias
  return label


def gaussian_label_function(target_bb,
                            sigma_factor,
                            kernel_sz,
                            feat_sz,
                            image_sz,
                            end_pad_if_even=True,
                            density=False,
                            uni_bias=0,
                            hw_sigma=False):
  """Construct Gaussian label function.
    args: 
        target_bb: shape [N,4] in the format (x,y,w,h)
        kernel_sz, shape [2] indicates keneral[h,w]
        feat_sz, shape [2] indicates feature [h,w]
        image_sz, [h, w]
    return:
        label_map: [N,H,W]
    """

  if isinstance(kernel_sz, (float, int)):
    kernel_sz = (kernel_sz, kernel_sz)
  if isinstance(feat_sz, (float, int)):
    feat_sz = (feat_sz, feat_sz)
  if isinstance(image_sz, (float, int)):
    image_sz = (image_sz, image_sz)

  image_sz = torch.Tensor(image_sz)
  feat_sz = torch.Tensor(feat_sz)

  target_center = target_bb[:, 0:2] + 0.5 * target_bb[:, 2:4]
  target_center_norm = (target_center.flip((1)) - image_sz / 2) / image_sz

  center = feat_sz * target_center_norm + 0.5 * \
           torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

  if hw_sigma:
    sigma = sigma_factor * target_bb[:, 2:4].flip(1)
  else:
    sigma = sigma_factor * feat_sz.prod().sqrt().item()

  if end_pad_if_even:
    end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
  else:
    end_pad = (0, 0)
  gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
  if density:
    sz = (feat_sz + torch.Tensor(end_pad)).prod()
    label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
  else:
    label = gauss_label + uni_bias
  return label


def gauss_density_centered(x, std):
  """Evaluate the probability density of a Gaussian centered at zero.
    args:
        x - Samples.
        std - List of standard deviations
    """
  return torch.exp(-0.5 * (x / std)**2) / (math.sqrt(2 * math.pi) * std)


def gmm_density_centered(x, std):
  """Evaluate the probability density of a GMM centered at zero.
    args:
        x - Samples. Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
        std - Tensor of standard deviations
    """
  if x.dim() == std.dim() - 1:
    x = x.unsqueeze(-1)
  elif not (x.dim() == std.dim() and x.shape[-1] == 1):
    raise ValueError('Last dimension must be the gmm stds.')
  return gauss_density_centered(x, std).prod(-2).mean(-1)


def sample_gmm_centered(std, num_samples=1):
  """Sample from a GMM distribution centered at zero:
    args:
        std - Tensor of standard deviations
        num_samples - number of samples
    """
  num_components = std.shape[-1]
  num_dims = std.numel() // num_components

  std = std.view(1, num_dims, num_components)

  # Sample component ids
  k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
  std_samp = std[0, :, k].t()

  # Sample
  x_centered = std_samp * torch.randn(num_samples, num_dims)
  prob_dens = gmm_density_centered(x_centered, std)

  return x_centered, prob_dens


def sample_gmm(mean, std, num_samples=1):
  """Sample from a GMM distribution:
    args:
        mean - a single mean vector
        std - Tensor of standard deviations
        num_samples - number of samples
    """
  num_dims = mean.numel()
  num_components = std.shape[-1]

  mean = mean.view(1, num_dims)
  std = std.view(1, -1, num_components)

  # Sample component ids
  k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
  std_samp = std[0, :, k].t()

  # Sample
  x_centered = std_samp * torch.randn(num_samples, num_dims)
  x = x_centered + mean
  prob_dens = gmm_density_centered(x_centered, std)

  return x, prob_dens


def sample_box_gmm(mean_box,
                   proposal_sigma,
                   gt_sigma=None,
                   num_samples=1,
                   add_mean_box=False):
  """Sample boxes from a Gaussian mixture model.
    args:
        mean_box - Center (or mean) bounding box
        proposal_sigma - List of standard deviations for each Gaussian
        gt_sigma - Standard deviation of the ground truth distribution
        num_samples - Number of sampled boxes
        add_mean_box - Also add mean box as first element

    returns:
        proposals, proposal density and ground truth density for all samples
    """
  center_std = torch.Tensor([s[0] for s in proposal_sigma])
  sz_std = torch.Tensor([s[1] for s in proposal_sigma])
  std = torch.stack([center_std, center_std, sz_std, sz_std])

  mean_box = mean_box.view(1, 4)
  sz_norm = mean_box[:, 2:].clone()

  # Sample boxes
  proposals_rel_centered, proposal_density = sample_gmm_centered(
      std, num_samples)

  # Add mean and map back
  mean_box_rel = rect_to_rel(mean_box, sz_norm)
  proposals_rel = proposals_rel_centered + mean_box_rel
  proposals = rel_to_rect(proposals_rel, sz_norm)

  if gt_sigma is None or gt_sigma[0] == 0 and gt_sigma[1] == 0:
    gt_density = torch.zeros_like(proposal_density)
  else:
    std_gt = torch.Tensor([gt_sigma[0], gt_sigma[0], gt_sigma[1],
                           gt_sigma[1]]).view(1, 4)
    gt_density = gauss_density_centered(proposals_rel_centered, std_gt).prod(-1)

  if add_mean_box:
    proposals = torch.cat((mean_box, proposals))
    proposal_density = torch.cat((torch.Tensor([-1]), proposal_density))
    gt_density = torch.cat((torch.Tensor([1]), gt_density))

  return proposals, proposal_density, gt_density


def sample_target_brpadding(im,
                            target_bb,
                            search_area_factor,
                            output_sz=None,
                            mask=None):
  """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
  if not isinstance(target_bb, list):
    x, y, w, h = target_bb.tolist()
  else:
    x, y, w, h = target_bb
  # Crop image
  crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

  if crop_sz < 1:
    raise Exception('Too small bounding box.')

  x1 = round(x + 0.5 * w - crop_sz * 0.5)
  x2 = x1 + crop_sz

  y1 = round(y + 0.5 * h - crop_sz * 0.5)
  y2 = y1 + crop_sz

  x1_pad = max(0, -x1)
  x2_pad = max(x2 - im.shape[1] + 1, 0)

  y1_pad = max(0, -y1)
  y2_pad = max(y2 - im.shape[0] + 1, 0)

  top_left_coord = (max(0, y1), max(0, x1))

  # Crop target
  im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
  if mask is not None:
    mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

  # Pad
  im_crop_padded = cv.copyMakeBorder(im_crop, 0, y1_pad + y2_pad, 0,
                                     x1_pad + x2_pad, cv.BORDER_CONSTANT)
  # deal with attention mask
  H, W, _ = im_crop_padded.shape
  att_mask = np.ones((H, W))
  end_x, end_y = -x2_pad - x1_pad, -y2_pad - y1_pad
  if y1_pad + y2_pad == 0:
    end_y = None
  if x1_pad + x2_pad == 0:
    end_x = None
  att_mask[:end_y, :end_x] = 0
  if mask is not None:
    mask_crop_padded = F.pad(mask_crop,
                             pad=(0, x1_pad + x2_pad, 0, y1_pad + y2_pad),
                             mode='constant',
                             value=0)

  if output_sz is not None:
    resize_factor = output_sz / crop_sz
    im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
    att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
    if mask is None:
      return im_crop_padded, resize_factor, top_left_coord, att_mask
    mask_crop_padded = \
    F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
    return im_crop_padded, resize_factor, top_left_coord, att_mask, mask_crop_padded

  else:
    if mask is None:
      return im_crop_padded, att_mask.astype(np.bool_), 1.0
    return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def jittered_brpadding_crop(frames,
                            box_extract,
                            box_gt,
                            search_area_factor,
                            output_sz,
                            masks=None):
  """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

  if masks is None:
    crops_resize_factors = [
        sample_target_brpadding(f, a, search_area_factor, output_sz)
        for f, a in zip(frames, box_extract)
    ]
    frames_crop, resize_factors, top_left_coords, att_mask = zip(
        *crops_resize_factors)
    masks_crop = None
  else:
    crops_resize_factors = [
        sample_target_brpadding(f, a, search_area_factor, output_sz, m)
        for f, a, m in zip(frames, box_extract, masks)
    ]
    frames_crop, resize_factors, top_left_coords, att_mask, masks_crop = zip(
        *crops_resize_factors)
  # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
  crop_sz = torch.Tensor([output_sz, output_sz])

  # find the bb location in the crop
  box_crop = [
      torch.tensor([box[0] - tl[1], box[1] - tl[0], box[2], box[3]]) * rf
      for box, tl, rf in zip(box_gt, top_left_coords, resize_factors)
  ]  # (x1,y1,w,h) list of tensors

  return frames_crop, box_crop, att_mask, masks_crop


def get_size_with_aspect_ratio(image_size, size=800, max_size=1440):
  h, w = image_size
  if max_size is not None:
    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))
    if max_original_size / min_original_size * size > max_size:
      size = int(round(max_size * min_original_size / max_original_size))

  if (w <= h and w == size) or (h <= w and h == size):
    return (h, w)

  if w < h:
    ow = size
    oh = int(size * h / w)
  else:
    oh = size
    ow = int(size * w / h)

  return (oh, ow)


def jitter_boxes(boxes, scale=0.2, center=2.0):
  """ Jitter the input boxes
      args:
          boxes - input bounding box in cxcywh format [N, 4]
          scale - box jitter size
          center - box jiter center

      returns:
          torch.Tensor - jittered box
      """

  jittered_size = boxes[..., 2:4] * torch.exp(
      torch.randn(*boxes.shape[:-1], 2).to(boxes.device)) * scale
  max_offset = jittered_size.prod(-1).sqrt() * center
  jittered_center = boxes[..., :2] + max_offset[..., None] * (
      torch.rand(*boxes.shape[:-1], 2).to(boxes.device) - 0.5)

  return torch.cat((jittered_center, jittered_size), dim=-1)


def generate_proposals(boxes,
                       proposal_method='default',
                       num_proposals=1,
                       in_format='xyxy',
                       out_format='xyxy',
                       iou_thresh=0.05):
  """ Generates proposals by adding noise to the input box
      args:
          boxes - input boxes in absolute format
          proposal_method - ways to generate proposals
          num_proposals - number of proposal for each box
          in_format - boxes format 'xyxy'/'xywh'
          iou_thresh - minumum iou threshold between proposals and given boxes

      returns:
          torch.Tensor -  proposals with the same shape as inputs
      """
  # Generate proposals
  new_boxes = box_convert(boxes.reshape(-1, 4), in_format,
                          'xywh')  ## convert xywh for processing
  sigma_factor = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
  output_proposals = []

  for box in new_boxes:
    if proposal_method == 'default':
      proposals = torch.zeros((num_proposals, 4), dtype=box.dtype, device=box.device)
      for i in range(num_proposals):
        proposals[i, :], _ = perturb_box(box,
                                         min_iou=iou_thresh,
                                         sigma_factor=sigma_factor)
    elif proposal_method == 'gmm':
      proposals, _, _ = sample_box_gmm(box, iou_thresh, num_samples=num_proposals)
    else:
      raise ValueError('Unknown proposal method.')
    output_proposals.append(box_convert(proposals, 'xywh', out_format))
  return torch.stack(output_proposals)


def normalize_boxes(boxes, image_sz, in_format='xyxy', out_format='xyxy', reverse=False):
  """[summary]

  Args:
      boxes ([..., 4]): [description]
      image_sz ([..., 2]): [H W]
      in_format (str, optional): [description]. Defaults to 'xyxy'.
      out_format (str, optional): [description]. Defaults to 'xywh'.
      reverse (bool, optional): [description]. Defaults to False.

  Returns:
      [type]: [description]
  """
  if isinstance(image_sz, torch.Tensor):
    wh = image_sz.flip(-1).detach().to(boxes.device)
  elif isinstance(image_sz, (list, tuple)):
    wh = torch.tensor(image_sz).flip(-1).to(boxes.device)
  else:
    raise TypeError('image shape not supported')
  norm_box = torch.cat([wh, wh], dim=-1)
  assert norm_box.dim() == 1 or boxes.dim() - norm_box.dim() <= 1
  if norm_box.dim() > 1 and boxes.dim() - norm_box.dim() == 1:
    norm_box = norm_box.unsqueeze(-2)
  if in_format != out_format:
    boxes = box_convert(boxes, in_format, out_format)
  if reverse:
    return boxes * norm_box
  return boxes / norm_box
