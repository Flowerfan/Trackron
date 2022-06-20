import numpy as np
from pathlib import Path
from trackron.structures import Sequence, SequenceList
from .base_dataset import BaseDataset
from .load_text import load_text
import os


class GOT10KDataset(BaseDataset):
  """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

  def __init__(self, data_root: Path, split: str):
    super().__init__()
    # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
    self.base_path = data_root / split

    self.sequence_list = self._get_sequence_list(split)
    self.split = split

  def get_sequence_list(self):
    return SequenceList(
        [self._construct_sequence(s) for s in self.sequence_list])

  def __getitem__(self, index):
    seq = self._construct_sequence(self.sequence_list[index])
    return seq

  def _construct_sequence(self, sequence_name):
    anno_path = self.base_path / sequence_name / 'groundtruth.txt'
    ground_truth_rect = load_text(str(anno_path),
                                  delimiter=',',
                                  dtype=np.float64)

    frames_path = self.base_path / sequence_name
    frames_list = [frame for frame in frames_path.glob("*.jpg")]
    frames_list = list(map(str, sorted(frames_list, key=lambda x: int(x.stem))))
    # frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
    # frame_list.sort(key=lambda f: int(f[:-4]))
    # frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

    return Sequence(sequence_name, frames_list, 'got10k',
                    ground_truth_rect.reshape(-1, 4))

  def __len__(self):
    return len(self.sequence_list)

  def _get_sequence_list(self, split):
    with open('{}/list.txt'.format(self.base_path)) as f:
      sequence_list = f.read().splitlines()

    if split == 'ltrval':
      with open('{}/got10k_val_split.txt'.format(
          self.env_settings.dataspec_path)) as f:
        seq_ids = f.read().splitlines()

      sequence_list = [sequence_list[int(x)] for x in seq_ids]
    return sequence_list
