import math
import torch.nn as nn
from collections import OrderedDict
from .base import Backbone
from .build import BACKBONE_REGISTRY
from trackron.models.layers import get_norm, ShapeSpec


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  """3x3 convolution with padding"""
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
               norm="BN"):
    super(BasicBlock, self).__init__()
    self.norm = norm
    self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
    self.bn1 = get_norm(norm, planes)

    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes, dilation=dilation)
    self.bn2 = get_norm(norm, planes)

    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    if self.bn1 is not None:
      out = self.bn1(out)

    out = self.relu(out)

    out = self.conv2(out)
    if self.bn2 is not None:
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
    self.conv2 = nn.Conv2d(planes,
                           planes,
                           kernel_size=3,
                           stride=stride,
                           padding=dilation,
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


class ResNet(Backbone):
  """ ResNet network module. Allows extracting specific feature blocks."""

  def __init__(self,
               block,
               num_layers,
               output_layers,
               num_classes=1000,
               inplanes=64,
               dilation_factor=1,
               frozen_stages=-1,
               norm="BN"):
    super(ResNet, self).__init__()
    self.inplanes = inplanes
    self.frozen_stages = frozen_stages
    self.norm = norm
    self.output_layers = output_layers if output_layers else ['layer4']
    self.conv1 = nn.Conv2d(3,
                           inplanes,
                           kernel_size=7,
                           stride=2,
                           padding=3,
                           bias=False)
    self.bn1 = get_norm(norm, inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    layer_strides = [1 + (dilation_factor < l) for l in (1, 2, 4, 8)]
    out_feature_strides = {'conv1': 4}
    out_feature_channels = {'conv1': inplanes}
    in_channel = inplanes
    for i in range(4):
      in_channel = inplanes * (2**i)
      layer = self._make_layer(block,
                               in_channel,
                               num_layers[i],
                               stride=layer_strides[i])
      setattr(self, f'layer{i+1}', layer)
      out_feature_channels[f'layer{i+1}'] = in_channel * block.expansion
      out_feature_strides[f'layer{i+1}'] = 4 * (2**i)
      if output_layers[-1] == f'layer{i+1}':
        break

    self._out_feature_strides = out_feature_strides
    self._out_feature_channels = out_feature_channels

    # self.avgpool = nn.AvgPool2d(7, stride=1)
    if 'fc' in output_layers:
      self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
      self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

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

  def out_feature_strides(self, layer=None):
    if layer is None:
      return self._out_feature_strides
    else:
      return self._out_feature_strides[layer]

  def out_feature_channels(self, layer=None):
    if layer is None:
      return self._out_feature_channels
    else:
      return self._out_feature_channels[layer]

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
          get_norm(self.norm, planes * block.expansion),
      )

    layers = []
    layers.append(
        block(self.inplanes, planes, stride, downsample, dilation=dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def _add_output_and_check(self, name, x, outputs, output_layers):
    if name in output_layers:
      outputs[name] = x
    return len(output_layers) == len(outputs)

  def forward(self, x, output_layers=None, remove_first_pool=False):
    """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
    outputs = OrderedDict()

    if output_layers is None:
      output_layers = self.output_layers

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    if self._add_output_and_check('conv1', x, outputs, output_layers):
      return outputs

    if not remove_first_pool:
      x = self.maxpool(x)

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

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    if self._add_output_and_check('fc', x, outputs, output_layers):
      return outputs

    if len(output_layers) == 1 and output_layers[0] == 'default':
      return x

    raise ValueError('output_layer is wrong.')

  def output_shape(self):
    return {
        name: ShapeSpec(channels=self._out_feature_channels[name],
                        stride=self._out_feature_strides[name])
        for name in self.output_layers
    }

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


def resnet_baby(output_layers=None, pretrained=False, inplanes=16, **kwargs):
  """Constructs a ResNet-18 model.
    """

  if output_layers is None:
    output_layers = ['default']
  else:
    for l in output_layers:
      if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
        raise ValueError('Unknown layer: {}'.format(l))

  model = ResNet(BasicBlock, [2, 2, 2, 2],
                 output_layers,
                 inplanes=inplanes,
                 **kwargs)

  if pretrained:
    raise NotImplementedError
  return model


@BACKBONE_REGISTRY.register()
def resnet18(cfg):
  """Constructs a ResNet-18 model.
    """
  output_layers = cfg.MODEL.BACKBONE.OUTPUT_LAYERS
  pretrained = cfg.MODEL.BACKBONE.PRETRAIN

  if output_layers is None:
    output_layers = ['default']
  else:
    for l in output_layers:
      if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
        raise ValueError('Unknown layer: {}'.format(l))

  model = ResNet(BasicBlock, [2, 2, 2, 2],
                 output_layers,
                 frozen_stages=cfg.MODEL.BACKBONE.FROZEN_STAGES,
                 norm=cfg.MODEL.BACKBONE.NORM)

  return model


@BACKBONE_REGISTRY.register()
def resnet50(cfg):
  """Constructs a ResNet-50 model.
    """
  output_layers = cfg.MODEL.BACKBONE.OUTPUT_LAYERS
  pretrained = cfg.MODEL.BACKBONE.PRETRAIN
  if output_layers is None:
    output_layers = ['default']
  else:
    for l in output_layers:
      if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
        raise ValueError('Unknown layer: {}'.format(l))

  model = ResNet(Bottleneck, [3, 4, 6, 3],
                 output_layers,
                 frozen_stages=cfg.MODEL.BACKBONE.FROZEN_STAGES,
                 norm=cfg.MODEL.BACKBONE.NORM)
  #   if pretrained:
  #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


@BACKBONE_REGISTRY.register()
def resnet101(cfg):
  """Constructs a ResNet-101 model.
    """
  output_layers = cfg.MODEL.BACKBONE.OUTPUT_LAYERS
  pretrained = cfg.MODEL.BACKBONE.PRETRAIN

  if output_layers is None:
    output_layers = ['default']
  else:
    for l in output_layers:
      if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
        raise ValueError('Unknown layer: {}'.format(l))

  model = ResNet(Bottleneck, [3, 4, 23, 3],
                 output_layers,
                 frozen_stages=cfg.MODEL.BACKBONE.FROZEN_STAGES,
                 norm=cfg.MODEL.BACKBONE.NORM)
  return model
