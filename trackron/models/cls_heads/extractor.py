import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
from trackron.models.layers.normalization import InstanceL2Norm
from trackron.models.layers.transform import InterpCat

from fvcore.common.registry import Registry



EXTRACTOR_REGISTRY = Registry("EXTRACTOR")
EXTRACTOR_REGISTRY.__doc__ = """
Registry for Box Regression head, which extract box head
The registered object must be a callable that accepts two arguments:

1. A :class:`tracker.config.CfgNode`
2. A :class:`tracker.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


@EXTRACTOR_REGISTRY.register()
def residual_bottleneck(cfg):
  feature_dim = cfg.MODEL.DIMP.EXTRACTOR.FEATURE_DIM
  num_blocks = cfg.MODEL.DIMP.EXTRACTOR.NUM_BLOCKS
  l2norm = cfg.MODEL.DIMP.EXTRACTOR.FEATURE_NORM
  final_conv = cfg.MODEL.DIMP.EXTRACTOR.FINAL_CONV
  out_dim = cfg.MODEL.DIMP.EXTRACTOR.OUT_DIM
  interp_cat = cfg.MODEL.DIMP.EXTRACTOR.INTERP_CAT
  filter_size = cfg.MODEL.DIMP.FILTER_SIZE
  final_relu = False
  final_pool = False
  norm_scale = math.sqrt(1.0 / (out_dim * filter_size * filter_size))
  # def residual_bottleneck(feature_dim=256, num_blocks=1, l2norm=True, final_conv=False, norm_scale=1.0, out_dim=None,
  #                         interp_cat=False, final_relu=False, final_pool=False):
  #     """Construct a network block based on the Bottleneck block used in ResNet."""
  if out_dim is None:
    out_dim = feature_dim
  feat_layers = []
  if interp_cat:
    feat_layers.append(InterpCat())
  for i in range(num_blocks):
    planes = feature_dim if i < num_blocks - \
        1 + int(final_conv) else out_dim // 4
    feat_layers.append(Bottleneck(4*feature_dim, planes))
  if final_conv:
    feat_layers.append(nn.Conv2d(4*feature_dim, out_dim,
                                 kernel_size=3, padding=1, bias=False))
    if final_relu:
      feat_layers.append(nn.ReLU(inplace=True))
    if final_pool:
      feat_layers.append(nn.MaxPool2d(
          kernel_size=3, stride=2, padding=1))
  if l2norm:
    feat_layers.append(InstanceL2Norm(scale=norm_scale))
  return nn.Sequential(*feat_layers)

def build_dimp_extractor(cfg):
  return EXTRACTOR_REGISTRY.get(cfg.MODEL.DIMP.EXTRACTOR.NAME)(cfg)
