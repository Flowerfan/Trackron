import os
import io
import json
import datetime
import contextlib
import logging
import numpy as np
from pathlib import Path
from itertools import chain
from collections import defaultdict
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
# from trackron.evaluation.utils.load_text import load_text
from trackron.structures import BoxMode, TAO, Sequence, SequenceList
from .base_dataset import BaseDataset
"""
This file contains functions to parse TAO-format annotations into dicts in "Tracker format".
"""

logger = logging.getLogger(__name__)


def nested_dict():
  return defaultdict(nested_dict)


class MOTDataset(BaseDataset):
  """ TAO dataset.

    Publication:
        TAO: Tracking Any Object Dataset
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

  def __init__(self,
               data_root: Path,
               split='test',
               version='17',
               mode='sot',
               public_detection=False):
    super().__init__()
    # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
    data_dir = split if split == 'test' else 'train'
    # data_dir = 'test'
    self.base_path = data_root / f'MOT{version}/{data_dir}'
    self.ann_path = data_root / f'MOT{version}/annotations/{split}.json'

    self.dataset = json.load(open(self.ann_path))
    self.sequence_list = self._get_sequence_list()
    self.init_anns()
    self.split = split
    self.public_detection = public_detection

  def init_anns(self):
    print("Creating MOT index.")

    self.vids = {x['id']: x for x in self.dataset['videos']}
    self.tracks = {x['id']: x for x in self.dataset['tracks']}
    self.cats = {x['id']: x for x in self.dataset['categories']}

    self.imgs = {}
    self.vid_img_map = defaultdict(list)
    for image in self.dataset['images']:
      self.imgs[image['id']] = image
      self.vid_img_map[image['video_id']].append(image)

    self.vid_track_map = defaultdict(list)
    for track in self.tracks.values():
      self.vid_track_map[track['video_id']].append(track)

    self.anns = {}
    self.img_ann_map = defaultdict(list)
    self.cat_img_map = defaultdict(list)
    self.track_ann_map = defaultdict(list)
    negative_anns = []
    self.has_gt = len(self.dataset.get("annotations", [])) >= 1

    for ann in self.dataset.get("annotations", []):
      # The category id is redundant given the track id, but we still
      # require it for compatibility with TAO tools.
      ann['bbox'] = [float(x) for x in ann['bbox']]
      if (ann['bbox'][0] < 0 or ann['bbox'][1] < 0 or ann['bbox'][2] <= 0 or
          ann['bbox'][3] <= 0):
        negative_anns.append(ann['id'])
      assert 'category_id' in ann, (f'Category id missing in annotation: {ann}')
      assert (ann['category_id'] == self.tracks[ann['track_id']]['category_id'])
      self.track_ann_map[ann['track_id']].append(ann)
      self.img_ann_map[ann["image_id"]].append(ann)
      self.cat_img_map[ann["category_id"]].append(ann["image_id"])
      self.anns[ann["id"]] = ann

    print("Index created.")

  def get_sequence_list(self):
    return SequenceList(
        [self._construct_sequence(s) for s in self.sequence_list])

  def init_public_detections(self, data, public_detections, start_frame_id=0, type='fairmot'):
    frame_dets = defaultdict(list)
    for detections in public_detections:
      frame_id = int(detections[0]) - 1
      if type == 'fairmot':
        det = detections[1:6]                                                         ### for fairmot detection
      else:
        det = detections[2:7]                                                       ### for public given detection
      frame_dets[frame_id].append(det)
    last_frame_id = 0
    for frame_id in sorted(frame_dets.keys()):
      frame_idx= frame_id - start_frame_id
      if frame_idx >= 0:
        data[frame_idx]['detections'] = np.stack(frame_dets[frame_id])
      if frame_idx - last_frame_id > 1:
        ### missing frame detections
        print('public detection could be wrong')
        for idx in range(last_frame_id+1, frame_id):
          data[idx]['detections'] = np.array([[0, 0, 1.0, 1.0, -1.0]])
      last_frame_id = frame_idx


  def _construct_sequence(self, vid):
    init_data = nested_dict()
    object_ids = []
    bboxes = {}
    gt_name, gt_file = None, None
    min_init_idx = 0

    vname = self.vids[vid]['name']
    frames_list = list(
        sorted([
            str(self.base_path / f['file_name']) for f in self.vid_img_map[vid]
        ]))
    # frames_list = [str(frame) for frame in sorted(frames_path.glob('*.jpg'))]
    min_index = min([image['frame_index'] for image in self.vid_img_map[vid]])

    #### external detection
    if self.public_detection:
      # pub_det_path = self.base_path / vname / 'det/det.txt'                            ### public
      pub_det_path = self.base_path.parent / 'dets/fairmot_det' / f'{vname}.txt'          ####fairmot
      pub_dets = np.loadtxt(pub_det_path, dtype=np.float32, delimiter=',')
      self.init_public_detections(init_data, pub_dets, min_index, type='fairmot')
    if self.has_gt:
      gt_name = f'gt_{self.split}.txt' if self.split in [
          'train_half', 'val_half'
      ] else 'gt.txt'
      gt_file = self.base_path / vname / 'gt' / gt_name
      for image_info in self.vid_img_map[vid]:
        image_id = image_info['id']
        frame_idx = image_info['frame_index'] - min_index
        init_data[frame_idx]['image'] = image_info
        init_data[frame_idx]['annotations'] = self.img_ann_map[image_id]
      for track in self.vid_track_map[vid]:
        track_id = track['id']
        object_ids += [track_id]
        ann = self.get_init_objects(track_id)
        image_id = ann['image_id']
        frame_idx = self.imgs[ann['image_id']]['frame_index'] - min_index
        # bboxes[track_id] = ann['bbox']
        init_data[frame_idx]['bbox'][track_id] = ann['bbox']

      min_init_idx = min(init_data.keys())
      if min_init_idx != 0:
        frames_list = frames_list[min_init_idx:]
        new_init_data = {idx - min_init_idx: v for idx, v in init_data.items()}
        init_data = new_init_data
    gt_bboxes = init_data
    return Sequence(vname.split('/')[-1],
                    frames_list,
                    'mot',
                    gt_bboxes,
                    init_data=init_data,
                    object_ids=object_ids,
                    multiobj_mode=True,
                    data_root=self.base_path,
                    gt_file=gt_file,
                    coco_path=self.ann_path,
                    tracking_mode='mot')

  def get_init_objects(self, track_id, init_type='first'):
    if init_type == 'first':
      return self.get_kth_annotation(track_id, k=0)
    elif init_type == 'biggest':
      return max(self.track_ann_map[track_id], key=lambda x: x['area'])
    else:
      raise NotImplementedError(f'Unsupported init type, {init_type}')

  def get_kth_annotation(self, track_id, k):
    """Return kth annotation for a track."""
    return sorted(self.track_ann_map[track_id], key=lambda x: x['image_id'])[k]

  def __len__(self):
    return len(self.sequence_list)

  def _get_sequence_list(self):
    sequence_list = [d['id'] for d in self.dataset['videos']]
    return sequence_list
