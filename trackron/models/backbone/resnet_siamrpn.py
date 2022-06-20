import math

import torch.nn as nn
from collections import OrderedDict
from .build import BACKBONE_REGISTRY
from .base import Backbone
from trackron.models.layers import get_norm

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes,
                   out_planes,
                   kernel_size=3,
                   stride=stride,
                   padding=dilation,
                   bias=False,
                   dilation=dilation)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               dilation=1,
               norm='BN'):
    super(BasicBlock, self).__init__()
    padding = 2 - stride

    if dilation > 1:
      padding = dilation

    dd = dilation
    pad = padding
    if downsample is not None and dilation > 1:
      dd = dilation // 2
      pad = dd

    self.conv1 = nn.Conv2d(inplanes,
                           planes,
                           stride=stride,
                           dilation=dd,
                           bias=False,
                           kernel_size=3,
                           padding=pad)
    self.bn1 = get_norm(norm, planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes, dilation=dilation)
    self.bn2 = get_norm(norm, planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               downsample=None,
               dilation=1,
               norm="BN"):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = get_norm(norm, planes)
    padding = 2 - stride
    if downsample is not None and dilation > 1:
      dilation = dilation // 2
      padding = dilation

    assert stride == 1 or dilation == 1, \
        "stride and dilation must have one equals to zero at least"

    if dilation > 1:
      padding = dilation
    self.conv2 = nn.Conv2d(planes,
                           planes,
                           kernel_size=3,
                           stride=stride,
                           padding=padding,
                           bias=False,
                           dilation=dilation)
    self.bn2 = get_norm(norm, planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = get_norm(norm, planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual

    out = self.relu(out)

    return out


class ResNetRPN(Backbone):

  def __init__(self,
               block,
               layers,
               output_layers,
               inplanes=64,
               dilation_factor=1,
               frozen_stages=-1,
               norm="BN"):
    super(ResNetRPN, self).__init__()
    self.inplanes = inplanes
    self.norm = norm
    self.frozen_stages = frozen_stages
    self.output_layers = output_layers if output_layers else ['layer4']
    self.conv1 = nn.Conv2d(
        3,
        inplanes,
        kernel_size=7,
        stride=2,
        padding=0,  # 3
        bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.bn1 = get_norm(norm, self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, inplanes, layers[0])
    self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                   dilation=2)  # 15x15, 7x7
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                   dilation=4)  # 7x7, 3x3

    out_feature_strides = {
        'conv1': 4,
        'layer1': 4,
        'layer2': 8,
        'layer3': 8,
        'layer4': 8
    }
    out_feature_channels = {
        'conv1': inplanes,
        'layer1': inplanes * 4,
        'layer2': inplanes * 8,
        'layer3': inplanes * 16,
        'layer4': inplanes * 32,
    }

    self._out_feature_strides = out_feature_strides
    self._out_feature_channels = out_feature_channels

    self.init_weights()
    self._freeze_stages()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _freeze_stages(self):
    if self.frozen_stages >= 0:
      self.conv1.eval()
      for param in self.conv1.parameters():
        param.requires_grad = False

    if self.frozen_stages >= 1:
      for i in range(1, self.frozen_stages + 1):
        m = getattr(self, "layer{}".format(i))
        m.eval()
        for param in m.parameters():
          param.requires_grad = False

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    dd = dilation
    if stride != 1 or self.inplanes != planes * block.expansion:
      if stride == 1 and dilation == 1:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes,
                      planes * block.expansion,
                      kernel_size=1,
                      stride=stride,
                      bias=False),
            get_norm(self.norm, planes * block.expansion),
        )
      else:
        if dilation > 1:
          dd = dilation // 2
          padding = dd
        else:
          dd = 1
          padding = 0
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes,
                      planes * block.expansion,
                      kernel_size=3,
                      stride=stride,
                      bias=False,
                      padding=padding,
                      dilation=dd),
            get_norm(self.norm, planes * block.expansion),
        )

    layers = []
    layers.append(
        block(self.inplanes,
              planes,
              stride,
              downsample,
              dilation=dilation,
              norm=self.norm))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, dilation=dilation))

    return nn.Sequential(*layers)

  def _add_output_and_check(self, name, x, outputs, output_layers):
    if name in output_layers:
      outputs[name] = x
    return len(output_layers) == len(outputs)

  def forward(self, x, output_layers=None):
    outputs = OrderedDict()
    if output_layers is None:
      output_layers = self.output_layers

    x = self.conv1(x)
    x = self.bn1(x)
    x_ = self.relu(x)
    x = self.maxpool(x_)

    if self._add_output_and_check('conv1', x, outputs, output_layers):
      return outputs

    x = self.layer1(x)
    if self._add_output_and_check('layer1', x, outputs, output_layers):
      return outputs

    x = self.layer2(x)
    if self._add_output_and_check('layer2', x, outputs, output_layers):
      return outputs
    x = self.layer3(x)
    if self._add_output_and_check('layer3', x, outputs, output_layers):
      return outputs
    x = self.layer4(x)
    if self._add_output_and_check('layer4', x, outputs, output_layers):
      return outputs



@BACKBONE_REGISTRY.register()
def resnet50_rpn(cfg):
  output_layers = cfg.MODEL.BACKBONE.OUTPUT_LAYERS
  if output_layers is None:
    output_layers = ['default']
  else:
    for l in output_layers:
      if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
        raise ValueError('Unknown layer: {}'.format(l))
  model = ResNetRPN(Bottleneck, [3, 4, 6, 3],
                    output_layers,
                    frozen_stages=cfg.MODEL.BACKBONE.FROZEN_STAGES,
                    norm=cfg.MODEL.BACKBONE.NORM)
  return model
