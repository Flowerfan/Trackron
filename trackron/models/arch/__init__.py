# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .dimp import DiMPNet
from .sequence_sot import SequenceSOT
from .siamese_mot import SiameseMOT
from .siamese_sot import SiameseSOT
from .utt import UnifiedTransformerTracker, UnifiedTransformerTracker2
from .detection_mot import DetectionMOT


__all__ = list(globals().keys())
