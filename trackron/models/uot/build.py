import torch
import torch.nn as nn
from fvcore.common.registry import Registry
from trackron.config import configurable


UOT_HEAD_REGISTRY = Registry("SOT_HEAD")  # noqa F401 isort:skip
UOT_HEAD_REGISTRY.__doc__ = """
Registry for unified object tracking head, i.e. uot head

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

class UOTHead(nn.Module):
  
  def __init__(self):
    super().__init__()
  
  

def build_uot_head(cfg, in_feature_channels):
    uot_head = UOT_HEAD_REGISTRY.get(cfg.NAME)(cfg, in_feature_channels)
    return uot_head

