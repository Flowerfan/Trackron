# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .lvis_v0_5_categories import LVIS_CATEGORIES as LVIS_V0_5_CATEGORIES
"""
Note:
For your custom dataset, there is no need to hard-code metadata anywhere in the code.
For example, for TAO-format dataset, metadata will be obtained automatically
when calling `load_lvis_json`. For other dataset, metadata may also be obtained in other ways
during loading.

However, we hard-coded metadata for a few common dataset here.
The only goal is to allow users who don't have these dataset to use pre-trained models.
Users don't have to download a lvis json (which contains metadata), in order to visualize a
TAO model (with correct class names and colors).
"""


def _get_lvis_instances_meta_v0_5():
  assert len(LVIS_V0_5_CATEGORIES) == 1230
  cat_ids = [k["id"] for k in LVIS_V0_5_CATEGORIES]
  assert min(cat_ids) == 1 and max(cat_ids) == len(
      cat_ids), "Category ids are not in [1, #categories], as expected"
  # Ensure that the category list is sorted by id
  lvis_categories = sorted(LVIS_V0_5_CATEGORIES, key=lambda x: x["id"])
  thing_classes = [k["synonyms"][0] for k in lvis_categories]
  meta = {"thing_classes": thing_classes}
  return meta




def _get_builtin_metadata(dataset_name):
  if dataset_name == "tao":
    return _get_lvis_instances_meta_v0_5()
  raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
