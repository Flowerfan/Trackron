from abc import abstractmethod
import torch
import torch.nn as nn
from fvcore.common.registry import Registry
from trackron.config import configurable

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole tracking model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
  model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg)
  model.to(torch.device(cfg.MODEL.DEVICE))
  return model


class BaseModel(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, data, mode='sot'):
    if mode == 'sot':
      return self.forward_sot(data)
    elif mode == 'mot':
      return self.forward_mot(data)
  
  @abstractmethod
  def forward_sot(self, data):
    raise NotImplementedError

  @abstractmethod
  def forward_mot(self, data):
    raise NotImplementedError
