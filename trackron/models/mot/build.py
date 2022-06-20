import torch
import torch.nn as nn
from fvcore.common.registry import Registry
from trackron.config import configurable


MOT_HEAD_REGISTRY = Registry("MOT_HEAD")  # noqa F401 isort:skip
MOT_HEAD_REGISTRY.__doc__ = """
Registry for single object tracking head, i.e. MOT head

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

class MOT(nn.Module):
  
  def __init__(self):
    super().__init__()
  
  

def build_mot_head(cfg, in_feature_channels):
    mot_head = MOT_HEAD_REGISTRY.get(cfg.NAME)(cfg, in_feature_channels)
    return mot_head

