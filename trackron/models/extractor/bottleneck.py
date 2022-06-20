import math
from torch import nn
from torchvision.models.resnet import Bottleneck
from trackron.config.config import configurable
from trackron.models.layers.normalization import InstanceL2Norm
from trackron.models.layers.transform import InterpCat
from .build import EXTRACTOR_REGISTRY




@EXTRACTOR_REGISTRY.register()
def residual_bottleneck(cfg):
  feature_dim = cfg.FEATURE_DIM
  num_blocks = cfg.NUM_BLOCKS
  out_dim = cfg.OUT_DIM
  norm_scale = math.sqrt(1 / (out_dim * cfg.FILTER_SIZE * cfg.FILTER_SIZE))
  feat_layers = []
  if cfg.INTERP_CAT:
    feat_layers.append(InterpCat())
  for i in range(num_blocks):
    planes = feature_dim if i < num_blocks - \
        1 + int(cfg.FINAL_CONV) else out_dim // 4
    feat_layers.append(Bottleneck(4*feature_dim, planes))
  if cfg.FINAL_CONV:
    feat_layers.append(nn.Conv2d(4*feature_dim, out_dim,
                                 kernel_size=3, padding=1, bias=False))
    if cfg.ACTIVATION:
      feat_layers.append(nn.ReLU(inplace=True))
    if cfg.POOL:
      feat_layers.append(nn.MaxPool2d(
          kernel_size=3, stride=2, padding=1))
  if cfg.FEATURE_NORM:
    feat_layers.append(InstanceL2Norm(scale=norm_scale))
  return nn.Sequential(*feat_layers)

class AdjustLayer(nn.Module):

  def __init__(self, in_channels, out_channels, center_size=7):
    super(AdjustLayer, self).__init__()
    self.downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )
    self.center_size = center_size

  def forward(self, x):
    x = self.downsample(x)
    if x.size(3) < 20:
      l = (x.size(3) - self.center_size) // 2
      r = l + self.center_size
      x = x[:, :, l:r, l:r]
    return x

@EXTRACTOR_REGISTRY.register()
class AdjustAllLayer(nn.Module):

  @configurable
  def __init__(self, *,in_channels, out_channels, center_size=7):
    super(AdjustAllLayer, self).__init__()
    self.num = len(out_channels)
    if self.num == 1:
      self.downsample = AdjustLayer(in_channels[0], out_channels[0],
                                    center_size)
    else:
      for i in range(self.num):
        self.add_module(
            'downsample' + str(i + 2),
            AdjustLayer(in_channels[i], out_channels[i], center_size))

  @classmethod
  def from_config(cls, cfg):
    return {"in_channels": cfg.IN_CHANNELS,
            "out_channels": cfg.OUT_CHANNELS,
            "center_size": cfg.CENTER_SIZE}

  def forward(self, features):
    if self.num == 1:
      return self.downsample(features)
    else:
      out = []
      for i in range(self.num):
        adj_layer = getattr(self, 'downsample' + str(i + 2))
        out.append(adj_layer(features[i]))
      return out
