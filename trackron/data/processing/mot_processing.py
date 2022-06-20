import torch
import numpy as np
from torchvision.ops import box_convert, clip_boxes_to_image
from trackron.utils.visdom import Visdom
from trackron.structures import TensorDict
from trackron.config import configurable
from trackron.data.datasets.trainsets.lvis_v0_5_categories import LVIS_CATEGORIES
from .base import SiameseBaseProcessing, stack_tensors
from .build import DATA_PROCESSING_REGISTRY
from ..transforms import transforms as tfm


@DATA_PROCESSING_REGISTRY.register()
class MOTProcessing(SiameseBaseProcessing):

  @configurable
  def __init__(self,
               search_area_factor,
               center_jitter_factor,
               scale_jitter_factor,
               crop_type='replicate',
               max_scale_change=None,
               mode='pair',
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
    self.center_jitter_factor = center_jitter_factor
    self.scale_jitter_factor = scale_jitter_factor
    self.crop_type = crop_type
    self.mode = mode
    self.max_scale_change = max_scale_change

    # self.visdom = Visdom(1, None, {})

  @classmethod
  def from_config(cls, cfg, training=False):
    search_area_factor = cfg.SEARCH.FACTOR
    output_sz = cfg.SEARCH.SIZE
    center_jitter_factor = {'search': cfg.SEARCH.CENTER_JITTER}
    scale_jitter_factor = {'search': cfg.SEARCH.SCALE_JITTER}
    mode = cfg.TRAIN.MODE
    min_sz = cfg.TRAIN.MIN_SIZE
    max_sz = cfg.TRAIN.MAX_SIZE
    sample_style = cfg.TRAIN.MIN_SIZE_SAMPLING
    if training:
      # transform = tfm.Transform(
      #     tfm.ResizeShortestEdge(min_sz, max_sz, sample_style=sample_style),
      #     tfm.ToGrayscale(probability=0.05), tfm.ToTensorAndJitter(0.2),
      #     tfm.RandomHorizontalFlip(0.5))
      joint_transform = tfm.Transform(
          tfm.RandomSelect(
              tfm.ResizeShortestEdge(min_sz, max_sz, sample_style=sample_style),
              tfm.Compose([
                  tfm.ResizeShortestEdge([800, 1000, 1200],
                                         None,
                                         sample_style=sample_style),
                  tfm.RandomSizeCrop(800, 1200),
                  tfm.ResizeShortestEdge(min_sz,
                                         max_sz,
                                         sample_style=sample_style)
              ])),)
      transform = tfm.Transform(tfm.ToTensor())
    else:
      joint_transform = None
      transform = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                tfm.ToTensorAndJitter(0.2))

    return {
        "search_area_factor": search_area_factor,
        "center_jitter_factor": center_jitter_factor,
        "scale_jitter_factor": scale_jitter_factor,
        "mode": mode,
        "joint_transform": joint_transform,
        "transform": transform,
    }

  def _get_jittered_boxes(self, boxes, mode):
    """ Jitter the input box
        args:
            box - input bounding box [N, 4] in xyxy format
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

    jittered_size = (boxes[:, 2:] - boxes[:, :2]) * \
        torch.exp(torch.randn(len(boxes), 2) * self.scale_jitter_factor[mode])
    max_offset = (jittered_size.prod(1).sqrt() *
                  torch.tensor(self.center_jitter_factor[mode]).float())
    jittered_center = (boxes[:, :2] + boxes[:, 2:]) * 0.5 + \
        max_offset[:, None] * (torch.rand(len(boxes), 2) - 0.5)

    return torch.cat((jittered_center - 0.5 * jittered_size,
                      jittered_center + 0.5 * jittered_size),
                     dim=1)

  def _valid_filter(self,
                    trackids_list,
                    boxes_list,
                    labels_list,
                    image_sz,
                    masks_list=None):
    new_trackids_list, new_box_list, new_label_list = [], [], []
    if masks_list is not None:
      new_mask_list = []
    else:
      new_mask_list = None

    height, width = image_sz
    for idx, (trackids, boxes,
              labels) in enumerate(zip(trackids_list, boxes_list, labels_list)):
      boxes = box_convert(boxes, 'xywh', 'xyxy')
      valid = (boxes[:, 0] <= width) & (boxes[:, 1] <= height) & (
          boxes[:, 2] > 0) & (boxes[:, 3] > 0) & (boxes[:, 2] > boxes[:, 0]) & (
              boxes[:, 3] > boxes[:, 1])
      # boxes = clip_boxes_to_image(boxes[valid], image_sz)
      boxes = boxes[valid]
      new_box_list += [boxes]
      new_label_list += [labels[valid]]
      new_trackids_list += [trackids[valid]]
      if masks_list is not None:
        new_mask_list += [masks_list[idx][valid]]
    return new_trackids_list, new_box_list, new_label_list, new_mask_list

  def __call__(self, data: TensorDict):
    """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'search_proposals', 'proposal_iou',
                'search_label' (optional), 'template_label' (optional), 'search_label_density' (optional), 'template_label_density' (optional)
        """
    template_boxes = torch.zeros((10, 4), dtype=torch.float32)
    search_boxes = torch.zeros((10, 4), dtype=torch.float32)
    valid = torch.zeros(10, dtype=torch.bool)
    share_track_ids = set([ann['track_id'] for ann in data['template_anno'][0]])
    for s in ['template', 'search']:
      assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
          "In pair mode, num template/search frames must be 1"

      # Add a uniform noise to the center pos
      # jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

      # self.visdom.register((crops[0], boxes[0]), 'Tracking', 1, 'Tracking')
      images_list = data[f'{s}_images']
      annotations = data[f'{s}_anno']
      trackids_list = [
          torch.tensor([ann['track_id']
                        for ann in anns], dtype=torch.long)
          for anns in annotations
      ]
      boxes_list = [
          torch.tensor([ann['bbox']
                        for ann in anns], dtype=torch.float32)
          for anns in annotations
      ]
      labels_list = [
          torch.tensor([ann['category_id']
                        for ann in anns], dtype=torch.long)
          for anns in annotations
      ]
      if 'mask' in annotations[0][0]:
        masks_list = [
            torch.tensor([ann['mask']
                          for ann in anns], dtype=bool)
            for anns in annotations
        ]
      else:
        masks_list = None

      new_roll = s == 'template'
      if masks_list is not None:
        images_list, boxes_list, masks_list = self.transform['joint'](
            image=images_list,
            bbox=boxes_list,
            mask=masks_list,
            joint=True,
            new_roll=new_roll)

        images_list, boxes_list, masks_list = self.transform[s](
            image=images_list, bbox=boxes_list, mask=masks_list, joint=True)
      else:
        images_list, boxes_list = self.transform['joint'](image=images_list,
                                                          bbox=boxes_list,
                                                          joint=True,
                                                          new_roll=new_roll)

        images_list, boxes_list = self.transform[s](image=images_list,
                                                    bbox=boxes_list,
                                                    joint=True)
      image_sz = images_list[0].shape[-2:]
      data[f'{s}_images'] = torch.stack(images_list)
      data[f'{s}_trackids'], data[f'{s}_boxes'], data[f'{s}_labels'], data[
          f'{s}_masks'] = self._valid_filter(trackids_list,
                                             boxes_list,
                                             labels_list,
                                             image_sz,
                                             masks_list=masks_list)
      for trackids in data[f'{s}_trackids']:
        share_track_ids = share_track_ids.intersection(set(trackids.tolist()))
      data[f'{s}_id_index'] = [{
          trackid.item(): idx for idx, trackid in enumerate(trackids)
      } for trackids in data[f'{s}_trackids']]

    temp_search_match_inds = []
    for temp_id_index, search_id_index in zip(data['template_id_index'],
                                              data['search_id_index']):
      matches = {}
      for track_id in share_track_ids:
        temp_index = temp_id_index.get(track_id, -1)
        search_index = search_id_index.get(track_id, -1)
        if temp_index != -1 and search_index != -1:
          # matches.append([temp_index, search_index])
          matches[temp_index] = search_index
      temp_search_match_inds.append(matches)
    data['matched_indices'] = temp_search_match_inds

    ## shared track ids accorss template and search images
    inds_list = [
        torch.tensor([id_index[track_id] for track_id in share_track_ids],
                     dtype=torch.long) for id_index in data['search_id_index']
    ]
    data['search_track_labels'] = [
        labels[inds] for labels, inds in zip(data['search_labels'], inds_list)
    ]
    data['search_track_boxes'] = [
        boxes[inds] for boxes, inds in zip(data['search_boxes'], inds_list)
    ]
    if data['search_masks'] is not None:
      data['search_track_masks'] = [
          masks[inds] for masks, inds in zip(data['search_masks'], inds_list)
      ]
    # if len(track_ids) > 100:
    #   track_ids = np.random.choice(list(track_ids), 100, replace=False)
    if len(share_track_ids) < 1:
      raise ValueError('not containing a same object in two frames')

    boxes = box_convert(data['search_boxes'][0], 'xyxy', 'xywh')
    image = (data['search_images'][0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
    masks = data['search_masks'][0].to(torch.uint8).numpy()

    # self.visdom = Visdom(debug=2)
    # self.visdom.register((image, boxes), 'Tracking', 1, 'Tracking')
    # self.visdom.register((image, boxes, masks), 'Tracking', 1, 'Tracking')
    # boxes = box_convert(data['template_boxes'][0], 'xyxy', 'xywh')
    # image = (data['template_images'][0].permute(1, 2, 0) * 255).numpy().astype(
    #     np.uint8)
    # self.visdom.register((image, boxes), 'Tracking', 1, 'Template')

    ### for producing crop area
    # jitter_boxes = self._get_jittered_boxes(data['search_track_boxes'], 'search')
    # data['jitter_boxes'] = jitter_boxes
    # search_area_boxes, search_target_boxes = self.get_search_box(image_sz, jitter_boxes, data['search_track_boxes'])
    # data['search_area_boxes'] = search_area_boxes
    # data['search_target_boxes'] = search_target_boxes
    # valid[:len(track_ids)] = True
    # data['track_valid'] = valid

    # if data["search_masks"] is None:
    #   data["search_masks"] = torch.zeros(
    #       (1, self.output_sz["search"], self.output_sz["search"]))

    # Prepare output
    if self.mode == 'sequence':
      data = data.apply(stack_tensors)
    else:
      data = data.apply(lambda x: x[0]
                        if isinstance(x, (list, tuple, torch.Tensor)) else x)

    return data


@DATA_PROCESSING_REGISTRY.register()
class MOTProcessingMultiClass(MOTProcessing):

  @configurable
  def __init__(self,
               search_area_factor,
               center_jitter_factor,
               scale_jitter_factor,
               crop_type='replicate',
               max_scale_change=None,
               mode='pair',
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
    super().__init__(search_area_factor,
                     center_jitter_factor,
                     scale_jitter_factor,
                     crop_type,
                     max_scale_change,
                     mode=mode,
                     *args,
                     **kwargs)
    self.categoires = LVIS_CATEGORIES
    self.class_id = self.extract_class_categoryid()

    # self.visdom = Visdom(1, None, {})

  def extract_class_categoryid(self):
    name_id_map = {}
    for cate in self.categoires:
      name_id_map[cate['name']] = cate['id'] - 1
      for name in cate.get("synonyms", []):
        name_id_map[name] = cate['id'] - 1
    return name_id_map

  def get_search_box(self, image_sz, jitter_box, search_boxes):
    cx, cy = ((jitter_box[:, :2] + jitter_box[:, 2:]) / 2.0).unbind(-1)
    sz = ((jitter_box[:, 2:] - jitter_box[:, :2]).prod(-1).sqrt() *
          self.search_area_factor)
    tx, ty = cx - sz / 2.0, cy - sz / 2.0
    search_areas = clip_boxes_to_image(
        torch.stack([tx, ty, tx + sz, ty + sz], dim=1), image_sz)
    x1, y1, x2, y2 = search_boxes.unbind(-1)
    tx, ty = search_areas[:, :2].unbind(-1)
    search_boxes = torch.stack([x1 - tx, y1 - ty, x2 - tx, y2 - ty], dim=-1)
    return search_areas, search_boxes

  def _valid_filter(self, trackids_list, boxes_list, labels_list, image_sz):
    new_trackids_list, new_box_list, new_label_list = [], [], []
    height, width = image_sz
    for trackids, boxes, labels in zip(trackids_list, boxes_list, labels_list):
      boxes = box_convert(boxes, 'xywh', 'xyxy')
      valid = (boxes[:, 0] <= width) & (boxes[:, 1] <= height) & (
          boxes[:, 2] > 0) & (boxes[:, 3] > 0) & (boxes[:, 2] > boxes[:, 0]) & (
              boxes[:, 3] > boxes[:, 1])
      # boxes = clip_boxes_to_image(boxes[valid], image_sz)
      boxes = boxes[valid]
      new_box_list += [boxes]
      new_label_list += [labels[valid]]
      new_trackids_list += [trackids[valid]]
    return new_trackids_list, new_box_list, new_label_list

  def _get_category_id(self, class_name):
    return self.class_id[class_name]

  def __call__(self, data: TensorDict):
    """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'search_proposals', 'proposal_iou',
                'search_label' (optional), 'template_label' (optional), 'search_label_density' (optional), 'template_label_density' (optional)
        """
    template_boxes = torch.zeros((10, 4), dtype=torch.float32)
    search_boxes = torch.zeros((10, 4), dtype=torch.float32)
    valid = torch.zeros(10, dtype=torch.bool)
    share_track_ids = set([ann['track_id'] for ann in data['template_anno'][0]])
    for s in ['template', 'search']:
      assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
          "In pair mode, num template/search frames must be 1"

      # Add a uniform noise to the center pos
      # jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

      # self.visdom.register((crops[0], boxes[0]), 'Tracking', 1, 'Tracking')
      images_list = data[f'{s}_images']
      annotations = data[f'{s}_anno']
      trackids_list = [
          torch.tensor([ann['track_id']
                        for ann in anns], dtype=torch.long)
          for anns in annotations
      ]
      boxes_list = [
          torch.tensor([ann['bbox']
                        for ann in anns], dtype=torch.float32)
          for anns in annotations
      ]
      labels_list = [
          torch.tensor(
              [self._get_category_id(ann['class_name']) for ann in anns],
              dtype=torch.long) for anns in annotations
      ]

      new_roll = s == 'template'
      images_list, boxes_list = self.transform['joint'](image=images_list,
                                                        bbox=boxes_list,
                                                        joint=True,
                                                        new_roll=new_roll)

      images_list, boxes_list = self.transform[s](image=images_list,
                                                  bbox=boxes_list,
                                                  joint=True)
      image_sz = images_list[0].shape[-2:]
      data[f'{s}_images'] = torch.stack(images_list)
      data[f'{s}_trackids'], data[f'{s}_boxes'], data[
          f'{s}_labels'] = self._valid_filter(trackids_list, boxes_list,
                                              labels_list, image_sz)
      for trackids in data[f'{s}_trackids']:
        share_track_ids = share_track_ids.intersection(set(trackids.tolist()))
      data[f'{s}_id_index'] = [{
          trackid.item(): idx for idx, trackid in enumerate(trackids)
      } for trackids in data[f'{s}_trackids']]

    ## shared track ids accorss template and search images
    inds_list = [
        torch.tensor([id_index[track_id] for track_id in share_track_ids],
                     dtype=torch.long) for id_index in data['search_id_index']
    ]
    data['search_track_boxes'] = [
        boxes[inds] for boxes, inds in zip(data['search_boxes'], inds_list)
    ]
    data['search_track_labels'] = [
        labels[inds] for labels, inds in zip(data['search_labels'], inds_list)
    ]
    # if len(track_ids) > 100:
    #   track_ids = np.random.choice(list(track_ids), 100, replace=False)
    # if len(track_ids) < 1:
    #   raise ValueError('not contain single object in two frames')

    # self.visdom = Visdom(debug=2)
    # boxes = box_convert(data['search_boxes'][0], 'xyxy', 'xywh')
    # image = (data['search_images'][0].permute(1, 2, 0) * 255).numpy().astype(
    #     np.uint8)
    # self.visdom.register((image, boxes), 'Tracking', 1, 'Tracking')
    # boxes = box_convert(data['template_boxes'][0], 'xyxy', 'xywh')
    # image = (data['template_images'][0].permute(1, 2, 0) * 255).numpy().astype(
    #     np.uint8)
    # self.visdom.register((image, boxes), 'Tracking', 1, 'Template')

    # Prepare output
    if self.mode == 'sequence':
      data = data.apply(stack_tensors)
    else:
      data = data.apply(lambda x: x[0]
                        if isinstance(x, (list, tuple, torch.Tensor)) else x)

    return data
