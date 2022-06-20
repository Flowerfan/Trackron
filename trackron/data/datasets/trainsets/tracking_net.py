import torch
import os
import json
import os.path
import numpy as np
import pandas
import random
from collections import OrderedDict

from trackron.utils.misc import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset


def list_sequences(root, set_ids):
  """ Lists all the videos in the input set_ids. Returns a list of tuples (set_id, video_name)

    args:
        root: Root directory to TrackingNet
        set_ids: Sets (0-11) which are to be used

    returns:
        list - list of tuples (set_id, video_name) containing the set_id and video_name for each sequence
    """
  sequence_list = []

  for s in set_ids:
    anno_dir = os.path.join(root, "TRAIN_" + str(s), "anno")

    sequences_cur_set = [(s, os.path.splitext(f)[0])
                         for f in os.listdir(anno_dir)
                         if f.endswith('.txt')]
    sequence_list += sequences_cur_set

  return sequence_list


class TrackingNet(BaseVideoDataset):
  """ TrackingNet dataset.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """

  def __init__(self,
               root,
               image_loader=jpeg4py_loader,
               set_ids=None,
               data_fraction=None):
    """
        args:
            root        - The path to the TrackingNet folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            set_ids (None) - List containing the ids of the TrackingNet sets to be used for training. If None, all the
                            sets (0 - 11) will be used.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
    super().__init__('TrackingNet', root, image_loader)

    if set_ids is None:
      set_ids = [i for i in range(12)]

    self.set_ids = set_ids

    # Keep a list of all videos. Sequence list is a list of tuples (set_id, video_name) containing the set_id and
    # video_name for each sequence
    self.sequence_list = list_sequences(self.root, self.set_ids)
    ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    self.sequence_resolution = json.load(
        open(os.path.join(ltr_path, 'data_specs',
                          'trackingnet_resolution.json')))

    if data_fraction is not None:
      self.sequence_list = random.sample(
          self.sequence_list, int(len(self.sequence_list) * data_fraction))

    self.seq_to_class_map, self.seq_per_class = self._load_class_info()

    # we do not have the class_lists for the tracking net
    self.class_list = list(self.seq_per_class.keys())
    self.class_list.sort()

  def _load_class_info(self):
    ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    class_map_path = os.path.join(ltr_path, 'data_specs',
                                  'trackingnet_classmap.txt')

    with open(class_map_path, 'r') as f:
      seq_to_class_map = {
          seq_class.split('\t')[0]: seq_class.rstrip().split('\t')[1]
          for seq_class in f
      }

    seq_per_class = {}
    for i, seq in enumerate(self.sequence_list):
      class_name = seq_to_class_map.get(seq[1], 'Unknown')
      if class_name not in seq_per_class:
        seq_per_class[class_name] = [i]
      else:
        seq_per_class[class_name].append(i)

    return seq_to_class_map, seq_per_class

  def get_name(self):
    return 'trackingnet'

  def has_class_info(self):
    return True

  def get_sequences_in_class(self, class_name):
    return self.seq_per_class[class_name]

  def _read_bb_anno(self, seq_id):
    set_id = self.sequence_list[seq_id][0]
    vid_name = self.sequence_list[seq_id][1]
    bb_anno_file = os.path.join(self.root, "TRAIN_" + str(set_id), "anno",
                                vid_name + ".txt")
    gt = pandas.read_csv(bb_anno_file,
                         delimiter=',',
                         header=None,
                         dtype=np.float32,
                         na_filter=False,
                         low_memory=False).values
    return torch.tensor(gt)

  def get_sequence_info(self, seq_id):
    bbox = self._read_bb_anno(seq_id)
    vid_name = self.sequence_list[seq_id][1]
    width, height = self.sequence_resolution[vid_name]
    valid = (bbox[:, 0] >= 0) & (bbox[:, 1] >= 0) & (bbox[:, 2] > 0) & (
        bbox[:, 3] > 0) & (bbox[:, 0] + bbox[:, 2] <=
                           width) & (bbox[:, 1] + bbox[:, 3] <= height)
    visible = valid.clone().byte()
    return {
        'bbox': bbox,
        'valid': valid,
        'visible': visible,
    }

  def _get_frame(self, seq_id, frame_id):
    set_id = self.sequence_list[seq_id][0]
    vid_name = self.sequence_list[seq_id][1]
    frame_path = os.path.join(self.root, "TRAIN_" + str(set_id), "frames",
                              vid_name,
                              str(frame_id) + ".jpg")
    return self.image_loader(frame_path)

  def _get_class(self, seq_id):
    seq_name = self.sequence_list[seq_id][1]
    return self.seq_to_class_map[seq_name]

  def get_class_name(self, seq_id):
    obj_class = self._get_class(seq_id)

    return obj_class

  def get_frames(self, seq_id, frame_ids, anno=None):
    frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

    if anno is None:
      anno = self.get_sequence_info(seq_id)

    anno_frames = {}
    for key, value in anno.items():
      anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

    obj_class = self._get_class(seq_id)

    object_meta = OrderedDict({
        'object_class_name': obj_class,
        'motion_class': None,
        'major_class': None,
        'root_class': None,
        'motion_adverb': None
    })

    return frame_list, anno_frames, object_meta

  def get_mot_frames(self, seq_id, frame_ids, anno=None):
    frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

    if anno is None:
      anno = self.get_sequence_info(seq_id)

    obj_class = self._get_class(seq_id)

    if anno is None:
      anno = self.get_sequence_info(seq_id)

    anns = []
    for fid in frame_ids:
      ann = {'track_id': seq_id, 'class_name': obj_class}
      for key, value in anno.items():
        ann[key] = value[fid].clone().tolist()
      anns += [[ann]]

    return frame_list, anns
