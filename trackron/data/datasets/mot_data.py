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
import traceback


@DATASET_REGISTRY.register()
class MOTDataset(torch.utils.data.Dataset):

  @configurable
  def __init__(self,
               datasets,
               p_datasets,
               samples_per_epoch,
               max_gap,
               num_template_frames,
               num_search_frames,
               processing=no_processing,
               frame_sample_mode='casual'):
    """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the template frames and the search frames.
            num_search_frames - Number of search frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'casual' or 'interval'. If 'casual', then the search frames are sampled in a casually,
                                otherwise randomly within the interval.
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
    self.num_template_frames = num_template_frames
    self.num_search_frames = num_search_frames
    self.processing = processing
    self.frame_sample_mode = frame_sample_mode

  @classmethod
  def from_config(cls, cfg, training=False):
    if training:
      datasets = get_trainsets(cfg.TRAIN.DATASET_NAMES,
                               cfg.ROOT)
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
        "num_template_frames": cfg.TEMPLATE.FRAMES,
        "num_search_frames": cfg.SEARCH.FRAMES,
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
    if num_ids == 0:
      return []
    if min_id is None or min_id < 0:
      min_id = 0
    if max_id is None or max_id > len(visible):
      max_id = len(visible)

    valid_ids = [i for i in range(min_id, max_id) if visible[i]]

    # No visible ids
    if len(valid_ids) == 0:
      return None

    return list(sorted(random.choices(valid_ids, k=num_ids)))

  def __getitem__(self, index):
    """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

    # Select a dataset
    dataset = random.choices(self.datasets, self.p_datasets)[0]
    is_video_dataset = dataset.is_video_sequence()
    retry_count = 0

    # Sample a sequence with enough visible frames
    enough_visible_frames = False
    while True:
      while not enough_visible_frames:
        # Sample a sequence
        seq_id = random.randint(0, dataset.get_num_sequences() - 1)

        # Sample frames
        seq_info_dict = dataset.get_sequence_info(seq_id)
        visible = seq_info_dict['visible']

        enough_visible_frames = visible.type(torch.int64).sum().item(
        ) > 2 * self.num_search_frames and len(visible) >= 10

        enough_visible_frames = enough_visible_frames or not is_video_dataset

      if is_video_dataset:
        search_frame_ids = None
        gap_increase = 0
        if self.frame_sample_mode == 'interval':
          # Sample frame numbers within interval defined by the first frame
          while search_frame_ids is None:
            base_frame_id = self._sample_visible_ids(visible, num_ids=1)
            extra_template_frame_ids = self._sample_visible_ids(
                visible,
                num_ids=self.num_template_frames - 1,
                min_id=base_frame_id[0] - self.max_gap - gap_increase,
                max_id=base_frame_id[0] + self.max_gap + gap_increase)
            if extra_template_frame_ids is None:
              gap_increase += 5
              continue
            template_frame_ids = base_frame_id + extra_template_frame_ids
            search_frame_ids = self._sample_visible_ids(
                visible,
                num_ids=self.num_search_frames,
                min_id=template_frame_ids[0] - self.max_gap - gap_increase,
                max_id=template_frame_ids[0] + self.max_gap + gap_increase)
            gap_increase += 5  # Increase gap until a frame is found

        elif self.frame_sample_mode == 'casual':
          # Sample search and template frames in a casual manner, i.e. search_frame_ids > template_frame_ids
          while search_frame_ids is None:
            base_frame_id = self._sample_visible_ids(
                visible,
                num_ids=1,
                min_id=self.num_template_frames - 1,
                max_id=len(visible) - self.num_search_frames)
            prev_frame_ids = self._sample_visible_ids(
                visible,
                num_ids=self.num_template_frames - 1,
                min_id=base_frame_id[0] - self.max_gap - gap_increase,
                max_id=base_frame_id[0])
            if prev_frame_ids is None:
              gap_increase += 5
              continue
            template_frame_ids = base_frame_id + prev_frame_ids
            search_frame_ids = self._sample_visible_ids(
                visible,
                min_id=template_frame_ids[0] + 1,
                max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                num_ids=self.num_search_frames)
            # Increase gap until a frame is found
            gap_increase += 5
      else:
        # In case of image dataset, just repeat the image to generate synthetic video
        template_frame_ids = [1] * self.num_template_frames
        search_frame_ids = [1] * self.num_search_frames

      template_frames, template_anno = dataset.get_mot_frames(
          seq_id, template_frame_ids)
      search_frames, search_anno = dataset.get_mot_frames(seq_id, search_frame_ids)

      # search_frames, search_anno = dataset.get_mot_frames(
      #     seq_id, search_frame_ids)

      data = TensorDict({
          'template_images': template_frames,
          'template_anno': template_anno,
          'search_images': search_frames,
          'search_anno': search_anno,
          'dataset': dataset.get_name(),
          # 'search_class': meta_obj_search.get('object_class_name')
      })
      try:
        return self.processing(data)
      except Exception as e:
        retry_count += 1
        enough_visible_frames = False
        traceback.print_exc()
        logger = logging.getLogger(__name__)
        logger.warning("failed retry data {}".format(retry_count))
        logger.warning(e)

      # return self.processing(data)
