import torch
import logging
import operator
import pickle
from pathlib import Path

from .loader import TrackingLoader
from .dataset_mapper import DatasetMapper
from .common import AspectRatioGroupedDataset, DatasetFromList
from .datasets import build_dataset
from .samplers import TrainingSampler, RepeatFactorTrainingSampler, InferenceSampler

from trackron.data.datasets.testsets import get_dataset
from trackron.structures import BoxMode
from trackron.utils.env import seed_all_rng
from trackron.utils.comm import get_world_size
from trackron.utils.logger import _log_api_usage
"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_batch_data_loader",
    "build_tracking_train_loader",
    "build_tracking_test_loader",
    "get_tracking_dataset_sequence",
    # "print_instances_class_histogram",
]


def build_batch_data_loader(dataset,
                            sampler,
                            total_batch_size,
                            *,
                            stack_dim=0,
                            aspect_ratio_grouping=False,
                            num_workers=0):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size)

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size,
            drop_last=True)  # drop_last so the batch always have the same size
        return TrackingLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            stack_dim=stack_dim,
            worker_init_fn=worker_init_reset_seed,
        )


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = build_dataset(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD)
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


def build_tracking_loader(cfg, *, sampler=None, training=False):
    dataset = build_dataset(cfg.DATASET, training=training)
    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        # logger = logging.getLogger(__name__)
        # logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD)
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))


#   sampler = TrainingSampler(dataset)
    world_size = get_world_size()
    total_batch_size = cfg.DATALOADER.BATCH_SIZE
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size)

    batch_size = total_batch_size // world_size
    stack_dim = cfg.DATALOADER.STACK_DIM
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size,
        drop_last=True)  # drop_last so the batch always have the same size
    collate_fn = trivial_batch_collator if cfg.DATALOADER.COLLATE_FN is not None else None
    loader = TrackingLoader('train',
                            dataset,
                            training=True,
                            batch_sampler=batch_sampler,
                            # batch_size=batch_size,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            collate_fn=collate_fn,
                            stack_dim=stack_dim,
                            worker_init_fn=worker_init_reset_seed)

    return loader


def get_tracking_dataset_sequence(data_root, data_names, versions, splits):
    """
    Load and prepare dataset sequence for instance detection/segmentation and semantic segmentation.

    Args:
        data_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(data_names, str):
        data_names = [data_names]
    assert len(data_names), data_names
    dataset_seqence_list = get_dataset(data_root, data_names, versions, splits)

    return dataset_seqence_list


def build_tracking_test_loader(cfg,
                               dataset_name=None,
                               sampler=None,
                               num_workers=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_tracking_dataset_sequence`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if dataset_name is None:
        dataset_names = cfg.DATASET.TEST.DATASET_NAMES
    else:
        dataset_names = [dataset_name]
    dataset = get_dataset(Path(cfg.DATASET.ROOT), dataset_names,
                                       cfg.DATASET.TEST.VERSIONS,
                                       cfg.DATASET.TEST.SPLITS)
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if sampler is None:
        sampler = InferenceSampler(len(dataset))
    # Always use 1 sequence per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler,
                                                          1,
                                                          drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)
