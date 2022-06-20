# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import MOT_HEAD_REGISTRY, build_mot_head  # isort:skip

# import all the meta_arch, so they will be registered
from .transtrack  import TransTrack
from .dfdetr_proposal import DFDETR_PROPOSAL
from .dfdetr import DFDETRHEAD


__all__ = list(globals().keys())

