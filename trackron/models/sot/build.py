import torch
import torch.nn as nn
from fvcore.common.registry import Registry
from trackron.config import configurable


SOT_HEAD_REGISTRY = Registry("SOT_HEAD")  # noqa F401 isort:skip
SOT_HEAD_REGISTRY.__doc__ = """
Registry for single object tracking head, i.e. sot head

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

class SOTHead(nn.Module):
  
  def __init__(self):
    super().__init__()
  
  

def build_sot_head(cfg, in_feature_channels):
    sot_head = SOT_HEAD_REGISTRY.get(cfg.NAME)(cfg, in_feature_channels)
    return sot_head

