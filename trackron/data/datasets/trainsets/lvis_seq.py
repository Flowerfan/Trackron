import os
import logging
from .base_video_dataset import BaseVideoDataset
from trackron.utils.misc import jpeg4py_loader
import torch
import random
from pycocotools.coco import COCO
from collections import OrderedDict
from .coco_seq import MSCOCOSeq


class LVISSeq(MSCOCOSeq):
  """ The LVIS dataset. LVIS is an image dataset. Thus, we treat each image as a sequence of length 1.

    Download the images along with annotations from https://www.lvisdataset.org/dataset. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - lvis_v1_train.json
            - images
                - train2017

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

  def __init__(self,
               root,
               image_loader=jpeg4py_loader,
               data_fraction=None,
               split="train",
               version="v0.5"):
    """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
            version - version of lvis dataset (v1)
        """
    # super().__init__('LVIS', root, image_loader)
    self.name = 'LVIS'
    self.root = root
    self.image_loader = image_loader

    self.img_pth = os.path.join(root, f'images/{split}2017')
    self.anno_path = os.path.join(
        root, 'annotations/lvis_{}_{}.json'.format(version, split))
    # Load the COCO set.
    self.coco_set = COCO(self.anno_path)
    for idx, k in enumerate(self.coco_set.anns):
      self.coco_set.anns[k]['track_id'] = idx

    self.cats = self.coco_set.cats

    self.class_list = self.get_class_list()

    self.sequence_list = self._get_sequence_list()

    if data_fraction is not None:
      self.sequence_list = random.sample(
          self.sequence_list, int(len(self.sequence_list) * data_fraction))
    self.seq_per_class = self._build_seq_per_class()

  def _get_sequence_list(self):
    ann_list = list(self.coco_set.anns.keys())
    seq_list = [a for a in ann_list]
    return seq_list

  def get_name(self):
    return 'lvis'

  def _get_sequence_resolution(self, anno):
    image_info = self.coco_set.loadImgs(anno['image_id'])
    height = image_info[0]['height']
    width = image_info[0]['width']
    return (height, width)

  def get_sequence_info(self, seq_id):
    anno = self._get_anno(seq_id)
    height, width = self._get_sequence_resolution(anno)

    bbox = torch.Tensor(anno['bbox']).view(1, 4)

    mask = torch.Tensor(self.coco_set.annToMask(anno)).unsqueeze(dim=0)

    # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
    valid = (bbox[:, 0] >= 0) & (bbox[:, 1] >= 0) & (bbox[:, 2] > 0) & (
        bbox[:, 3] > 0) & (bbox[:, 0] + bbox[:, 2] <=
                           width) & (bbox[:, 1] + bbox[:, 3] <=
                                     height) & (bbox[:, 2] * bbox[:, 3] > 900)
    visible = valid.clone().byte()

    return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}

  def _get_frames(self, seq_id):
    path = self.coco_set.loadImgs([ self.coco_set.anns[self.sequence_list[seq_id]]['image_id'] ])[0]['coco_url'].split('/')[-1]
    img = self.image_loader(os.path.join(self.img_pth, path))
    return img

  def get_mot_frames(self, seq_id=None, frame_ids=None, anno=None):
    image_info = self.coco_set.loadImgs(
        [self.coco_set.anns[self.sequence_list[seq_id]]['image_id']])[0]
    path = image_info['coco_url'].split('/')[-1]
    frame = self.image_loader(os.path.join(self.img_pth, path))
    frame_list = [frame.copy() for _ in frame_ids]

    # obj_class = self.get_class_name(seq_id)

    if anno is None:
      ann_ids = self.coco_set.getAnnIds(imgIds=[image_info['id']])
      anno = self.coco_set.loadAnns(ann_ids)
      # for idx, k in enumerate(anno):
      #   anno['track_id'] = k
    for ann in anno:
      ann.update({'class_name': self.cats[ann['category_id']]['name']})

    anns = [anno.copy() for _ in frame_ids]
    # anno = self.get_sequence_info(seq_id)

    return frame_list, anns
