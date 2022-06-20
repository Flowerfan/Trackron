import numpy as np
from trackron.structures import Sequence, SequenceList
from .base_dataset import BaseDataset
import os
from collections import defaultdict
import json


class TAODataset(BaseDataset):
  """ TAO dataset.

    Publication:
        TAO: Tracking Any Object Dataset
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

  def __init__(self, data_root, split='train'):
    super().__init__()
    # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
    self.base_path = data_root / 'frames'
    self.ann_path = data_root / 'annotations' / f'{split}.json'

    self.dataset = json.load(open(self.ann_path))
    self.init_anns()

    self.sequence_list = self._get_sequence_list()
    self.split = split

  def init_anns(self):
    print("Creating TAO index.")

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
    for ann in self.dataset["annotations"]:
      # The category id is redundant given the track id, but we still
      # require it for compatibility with COCO tools.
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

  def _construct_sequence(self, track_id):
    vid = self.tracks[track_id]['video_id']
    vname = self.vids[vid]['name']
    frames_path = '{}/{}'.format(self.base_path, vname)
    frame_path = self.base_path / vname
    frames_list = list(sorted([str(frame) for frame in frames_path.glob('*.jpg') ]))
    first_image_index = min(
        [image['frame_index'] for image in self.vid_img_map[vid]])
    init_data = defaultdict()
    object_ids = []
    bboxes = {}
    object_ids += [track_id]
    ann = self.get_init_objects(track_id)
    image_id = ann['image_id']
    object_first_image_index = self.imgs[ann['image_id']]['frame_index']
    # bboxes[track_id] = ann['bbox']
    init_data[0] = {'bbox': ann['bbox']}
    start_index = object_first_image_index - first_image_index
    frames_list = frames_list[start_index:]
    gt_bboxes = init_data
    sq_name = '{}_{}'.format(vname.split('/')[-1], track_id)
    return Sequence(sq_name, frames_list, 'tao', gt_bboxes, init_data=init_data)

  def get_init_objects(self, track_id, init_type='first'):
    if init_type == 'first':
      return self.get_kth_annotation(track_id, k=0)
    elif init_type == 'biggest':
      return max(self.track_ann_map[track_id], key=lambda x: x['area'])
    else:
      raise NotImplementedError(f'Unsupported init type, {init_type}')

  def get_kth_annotation(self, track_id, k):
    """Return kth annotation for a track."""
    return sorted(self.track_ann_map[track_id],
                  key=lambda x: self.imgs[x['image_id']]['frame_index'])[k]

  def __len__(self):
    return len(self.sequence_list)

  def _get_sequence_list(self):
    sequence_list = [track_id for track_id in self.tracks]
    return sequence_list
