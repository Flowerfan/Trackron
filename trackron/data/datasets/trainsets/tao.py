import os
from .base_video_dataset import BaseVideoDataset
from trackron.utils.misc import jpeg4py_loader
import torch
import random
from pycocotools.coco import COCO
from collections import OrderedDict, defaultdict


class TAO(BaseVideoDataset):
  """ The TAO dataset..

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
                - instances_train2017.json
            - images
                - train2014
                - train2017

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
  """

  def __init__(self,
               root,
               image_loader=jpeg4py_loader,
               data_fraction=None,
               split="train"):
    """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
            version - version of coco dataset (2014 or 2017)
    """
    super().__init__('TAO', root, image_loader)

    self.img_pth = os.path.join(root, 'frames')
    self.anno_path = os.path.join(root, 'annotations-1.1/{}.json'.format(split))

    # Load the COCO set.
    self.tao = COCO(self.anno_path)

    self.cats = self.tao.cats

    self.tracks = self.tao.dataset['tracks']

    self.class_list = self.get_class_list()

    self.video_to_images = self._get_seq_images()

    self.sequence_list = self._get_sequence_list()

    if data_fraction is not None:
      self.sequence_list = random.sample(
          self.sequence_list, int(len(self.sequence_list) * data_fraction))
    self.seq_per_class = self._build_seq_per_class()

  def _get_seq_images(self):
    video_to_images = defaultdict(list)
    for image in self.tao.dataset['images']:
      video_to_images[image['video_id']].append(image)
    return video_to_images

  def _get_sequence_list(self):
    seq_list = []
    for track in self.tracks:
      track_id = track['id']
      video_id = track['video_id']
      img_infos = self.video_to_images[video_id]
      img_ids = sorted([m['id'] for m in img_infos])
      seq_infos = {
          'num_imgs': len(img_ids),
          'img_ids': img_ids,
          'track_id': track_id,
          'anns': [],
          'video_id': video_id,
          'category_id': track['category_id']
      }
      for idx, img_id in enumerate(img_ids):
        anns = self.tao.loadAnns(self.tao.getAnnIds(imgIds=[img_id]))
        for ann in anns:
          if track_id == ann['track_id']:
            seq_infos['anns'] += [ann]
            break
        if len(seq_infos['anns']) == idx:
          seq_infos['anns'] += [{}]
      seq_list += [seq_infos]
    return seq_list

  def get_num_classes(self):
    return len(self.class_list)

  def get_name(self):
    return 'tao'

  def has_class_info(self):
    return True

  def get_class_list(self):
    class_list = []
    for cat_id in self.cats.keys():
      class_list.append(self.cats[cat_id]['name'])
    return class_list

  # def has_segmentation_info(self):
  #     return True

  def get_num_sequences(self):
    return len(self.sequence_list)

  def _build_seq_per_class(self):
    seq_per_class = {}
    for i, seq in enumerate(self.sequence_list):
      class_name = self.cats[seq['category_id']]['name']
      if class_name not in seq_per_class:
        seq_per_class[class_name] = [i]
      else:
        seq_per_class[class_name].append(i)

    return seq_per_class

  def get_sequences_in_class(self, class_name):
    return self.seq_per_class[class_name]

  def get_sequence_info(self, seq_id):
    seq_anns = self.sequence_list[seq_id]['anns']
    bbox = [ann.get('bbox', [0, 0, 0, 0]) for ann in seq_anns]
    bbox = torch.Tensor(bbox).view(-1, 4)

    valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
    visible = valid.clone().byte()

    return {'bbox': bbox, 'valid': valid, 'visible': visible}

  def _get_frame(self, frame_path):
    img = self.image_loader(os.path.join(self.img_pth, frame_path))
    return img

  def get_seq_imgIds(self, seq_id):
    img_ids = self.sequence_list[seq_id]['img_ids']
    return img_ids

  def get_class_name(self, seq_id):
    cat_dict_current = self.cats[self.sequence_list[seq_id]['category_id']]
    return cat_dict_current['name']

  def get_frames(self, seq_id=None, frame_ids=None, anno=None):
    frame_infos = self.tao.loadImgs(
        sorted(self.sequence_list[seq_id]['img_ids']))
    frame_list = [
        self._get_frame(frame_infos[fid]['file_name']) for fid in frame_ids
    ]
    obj_class = self.get_class_name(seq_id)

    if anno is None:
      anno = self.get_sequence_info(seq_id)

    anno_frames = {}
    for key, value in anno.items():
      anno_frames[key] = [value[f_id, ...] for f_id in frame_ids]

    object_meta = OrderedDict({
        'object_class_name': obj_class,
        'motion_class': None,
        'major_class': None,
        'root_class': None,
        'motion_adverb': None
    })

    return frame_list, anno_frames, object_meta

  def get_mot_frames(self, seq_id=None, frame_ids=None, anno=None):
    frame_infos = self.tao.loadImgs(
        sorted(self.sequence_list[seq_id]['img_ids']))
    frame_list = [
        self._get_frame(frame_infos[fid]['file_name']) for fid in frame_ids
    ]

    if anno is None:
      anno = [
          self.tao.loadAnns(self.tao.getAnnIds(imgIds=[frame_info['id']]))
          for frame_info in frame_infos
      ]
    annos = []
    for fid in frame_ids:
      frame_annos = []
      for ann in anno[fid]:
        ann.update({'class_name': self.cats[ann['category_id']]['name']})
        frame_annos += [ann]
      annos += [frame_annos]
        

    # annos = [anno[fid] for fid in frame_ids]

    # object_meta = OrderedDict({
    #     'object_class_name': obj_class,
    #     'motion_class': None,
    #     'major_class': None,
    #     'root_class': None,
    #     'motion_adverb': None
    # })
    return frame_list, annos

    # return frame_list, anno_frames, object_meta