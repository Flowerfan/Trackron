# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from trackron.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import  _get_builtin_metadata
# from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
# from .cityscapes_panoptic import register_all_cityscapes_panoptic
# from .coco import load_sem_seg, register_coco_instances
# from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
# from .lvis import get_lvis_instances_meta, register_lvis_instances
# from .pascal_voc import register_pascal_voc
from .tao import register_tao

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_TAO = {}
_PREDEFINED_SPLITS_TAO["tao"] = {
    "tao_train":
        ("tao/train", "tao/annotations/train.json"),
}



def register_all_tao(root):
  for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_TAO.items():
    for key, (image_root, json_file) in splits_per_dataset.items():
      # Assume pre-defined datasets live in `./datasets`.
      register_tao(
          key,
          _get_builtin_metadata(dataset_name),
          os.path.join(root, json_file)
          if "://" not in json_file else json_file,
          os.path.join(root, image_root),
      )




# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
  # Assume pre-defined datasets live in `./datasets`.
  _root = os.getenv("DETECTRON2_DATASETS", "datasets")
  register_all_tao(_root)
  # register_all_lvis(_root)
  # register_all_cityscapes(_root)
  # register_all_cityscapes_panoptic(_root)
  # register_all_pascal_voc(_root)
  # register_all_ade20k(_root)
