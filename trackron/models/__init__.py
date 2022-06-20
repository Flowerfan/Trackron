import torch
import torch.nn as nn
from trackron.config import configurable

from .arch import META_ARCH_REGISTRY, build_model
from .backbone import BACKBONE_REGISTRY
from .cls_heads import CLS_HEAD_REGISTRY
from .box_heads import BOX_HEAD_REGISTRY


