import random
import logging
import torch.utils.data
from pathlib import Path
from trackron.structures import TensorDict
from fvcore.common.registry import Registry
from trackron.config import configurable
from trackron.config import configurable
from .trainsets import get_trainsets
from ..processing import build_processing_class
from .build import DATASET_REGISTRY, no_processing


@DATASET_REGISTRY.register()
class VOSDataset(torch.utils.data.Dataset):
  """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames and ii) a set of test frames. The train frames, along with the
    ground-truth masks, are passed to the few-shot learner to obtain the target model parameters \tau. The test frames
    are used to compute the prediction accuracy.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is randomly
    selected from that dataset. A base frame is then sampled randomly from the sequence. The 'train frames'
    are then sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id], and the 'test frames'
    are sampled from the sequence from the range (base_frame_id, base_frame_id + max_gap] respectively. Only the frames
    in which the target is visible are sampled. If enough visible frames are not found, the 'max_gap' is increased
    gradually until enough frames are found. Both the 'train frames' and the 'test frames' are sorted to preserve the
    temporal order.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

  @configurable
  def __init__(self,
               datasets,
               p_datasets,
               samples_per_epoch,
               max_gap,
               num_test_frames,
               num_train_frames=1,
               processing=no_processing,
               p_reverse=None):
    """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            p_reverse - Probability that a sequence is temporally reversed
        """
    self.datasets = datasets

    # If p not provided, sample uniformly from all videos
    if p_datasets is None:
      p_datasets = [len(d) for d in self.datasets]

    # Normalize
    p_total = sum(p_datasets)
    self.p_datasets = [x / p_total for x in p_datasets]

    self.samples_per_epoch = samples_per_epoch
    self.max_gap = max_gap
    self.num_test_frames = num_test_frames
    self.num_train_frames = num_train_frames
    self.processing = processing

    self.p_reverse = p_reverse

  @classmethod
  def from_config(cls, cfg, training=False):
    if training:
      datasets = get_trainsets(cfg.TRAIN.DATASET_NAMES, cfg.ROOT)
      p_datasets = cfg.TRAIN.DATASETS_RATIO
    else:
      datasets = get_trainsets(cfg.VAL.DATASET_NAMES, cfg.ROOT)
      p_datasets = cfg.VAL.DATASETS_RATIO
    processing_class = build_processing_class(cfg, training)
    return {
        "datasets": datasets,
        "p_datasets": p_datasets,
        "samples_per_epoch": cfg.TRAIN.SAMPLE_PER_EPOCH,
        "max_gap": cfg.MAX_SAMPLE_INTERVAL,
        "num_search_frames": cfg.SEARCH.FRAMES,
        "num_template_frames": cfg.TEMPLATE.FRAMES,
        "processing": processing_class,
        "frame_sample_mode": cfg.SAMPLE_MODE
    }

  def __len__(self):
    return self.samples_per_epoch

  def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
    """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
    if min_id is None or min_id < 0:
      min_id = 0
    if max_id is None or max_id > len(visible):
      max_id = len(visible)

    valid_ids = [i for i in range(min_id, max_id) if visible[i]]

    # No visible ids
    if len(valid_ids) == 0:
      return None

    return random.choices(valid_ids, k=num_ids)

  def __getitem__(self, index):
    """
        args:
            index (int): Index (dataset index)

        returns:
            TensorDict - dict containing all the data blocks
        """

    # Select a dataset
    dataset = random.choices(self.datasets, self.p_datasets)[0]

    is_video_dataset = dataset.is_video_sequence()

    reverse_sequence = False
    if self.p_reverse is not None:
      reverse_sequence = random.random() < self.p_reverse

    # Sample a sequence with enough visible frames
    enough_visible_frames = False
    while not enough_visible_frames:
      # Sample a sequence
      seq_id = random.randint(0, dataset.get_num_sequences() - 1)

      # Sample frames
      seq_info_dict = dataset.get_sequence_info(seq_id)
      visible = seq_info_dict['visible']

      enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
          self.num_test_frames + self.num_train_frames)

      enough_visible_frames = enough_visible_frames or not is_video_dataset

    if is_video_dataset:
      train_frame_ids = None
      test_frame_ids = None
      gap_increase = 0

      # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
      while test_frame_ids is None:
        if gap_increase > 1000:
          raise Exception('Frame not found')

        if not reverse_sequence:
          base_frame_id = self._sample_visible_ids(
              visible,
              num_ids=1,
              min_id=self.num_train_frames - 1,
              max_id=len(visible) - self.num_test_frames)
          prev_frame_ids = self._sample_visible_ids(
              visible,
              num_ids=self.num_train_frames - 1,
              min_id=base_frame_id[0] - self.max_gap - gap_increase,
              max_id=base_frame_id[0])
          if prev_frame_ids is None:
            gap_increase += 5
            continue
          train_frame_ids = base_frame_id + prev_frame_ids
          test_frame_ids = self._sample_visible_ids(
              visible,
              min_id=train_frame_ids[0] + 1,
              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
              num_ids=self.num_test_frames)

          # Increase gap until a frame is found
          gap_increase += 5
        else:
          # Sample in reverse order, i.e. train frames come after the test frames
          base_frame_id = self._sample_visible_ids(
              visible,
              num_ids=1,
              min_id=self.num_test_frames + 1,
              max_id=len(visible) - self.num_train_frames - 1)
          prev_frame_ids = self._sample_visible_ids(
              visible,
              num_ids=self.num_train_frames - 1,
              min_id=base_frame_id[0],
              max_id=base_frame_id[0] + self.max_gap + gap_increase)
          if prev_frame_ids is None:
            gap_increase += 5
            continue
          train_frame_ids = base_frame_id + prev_frame_ids
          test_frame_ids = self._sample_visible_ids(
              visible,
              min_id=0,
              max_id=train_frame_ids[0] - 1,
              num_ids=self.num_test_frames)

          # Increase gap until a frame is found
          gap_increase += 5
    else:
      # In case of image dataset, just repeat the image to generate synthetic video
      train_frame_ids = [1] * self.num_train_frames
      test_frame_ids = [1] * self.num_test_frames

    # Sort frames
    train_frame_ids = sorted(train_frame_ids, reverse=reverse_sequence)
    test_frame_ids = sorted(test_frame_ids, reverse=reverse_sequence)

    all_frame_ids = train_frame_ids + test_frame_ids

    # Load frames
    all_frames, all_anno, meta_obj = dataset.get_frames(seq_id, all_frame_ids,
                                                        seq_info_dict)

    train_frames = all_frames[:len(train_frame_ids)]
    test_frames = all_frames[len(train_frame_ids):]

    train_anno = {}
    test_anno = {}
    for key, value in all_anno.items():
      train_anno[key] = value[:len(train_frame_ids)]
      test_anno[key] = value[len(train_frame_ids):]

    train_masks = train_anno['mask'] if 'mask' in train_anno else None
    test_masks = test_anno['mask'] if 'mask' in test_anno else None

    data = TensorDict({
        'train_images': train_frames,
        'train_masks': train_masks,
        'train_anno': train_anno['bbox'],
        'test_images': test_frames,
        'test_masks': test_masks,
        'test_anno': test_anno['bbox'],
        'dataset': dataset.get_name()
    })

    return self.processing(data)
