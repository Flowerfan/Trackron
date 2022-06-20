import numpy as np
import os
from trackron.structures import Sequence, SequenceList
from .base_dataset import BaseDataset
from .load_text import load_text


class TrackingNetDataset(BaseDataset):
  """ TrackingNet test set.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """

  def __init__(self, data_root):
    super().__init__()
    self.base_path = data_root

    sets = ['TEST']
    # if not isinstance(sets, (list, tuple)):
    #   if sets == 'TEST':
    #     sets = ['TEST']
    #   elif sets == 'TRAIN':
    #     sets = ['TRAIN_{}'.format(i) for i in range(5)]

    self.sequence_list = self._list_sequences(self.base_path, sets)

  def get_sequence_list(self):
    return SequenceList([
        self._construct_sequence(set_name, seq_name)
        for set_name, seq_name in self.sequence_list
    ])

  def _construct_sequence(self, set_name, sequence_name):
    anno_path = self.base_path / set_name / f'anno/{sequence_name}.txt'
    ground_truth_rect = load_text(str(anno_path),
                                  delimiter=',',
                                  dtype=np.float64,
                                  backend='numpy')

    frames_path = self.base_path / set_name / f'frames/{sequence_name}'
    frames_list = [frame for frame in frames_path.glob("*.jpg")]
    # frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
    frames_list = list(map(str, sorted(frames_list, key=lambda x: int(x.stem))))

    return Sequence(sequence_name, frames_list, 'trackingnet',
                    ground_truth_rect.reshape(-1, 4))

  def __len__(self):
    return len(self.sequence_list)

  def _list_sequences(self, root, set_ids):
    sequence_list = []

    for s in set_ids:
      anno_dir = root / s / "anno"
      sequences_cur_set = [(s, f.stem) for f in anno_dir.glob("*.txt")]

      sequence_list += sequences_cur_set

    return sequence_list
