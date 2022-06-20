import torch
import torch.nn as nn
from trackron.models.layers import ShapeSpec
from abc import ABCMeta, abstractmethod


class Backbone(nn.Module):
  """Base class for backbone networks. Handles freezing layers etc.
    args:
        frozen_layers  -  Name of layers to freeze. Either list of strings, 'none' or 'all'. Default: 'none'.
    """

  def __init__(self):
    super().__init__()

  @abstractmethod
  def forward(self):
    """
        Subclasses must override this method, but adhere to the same return type.

        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
    pass

  @property
  def size_divisibility(self) -> int:
    """
      Some backbones require the input height and width to be divisible by a
      specific integer. This is typically true for encoder / decoder type networks
      with lateral connection (e.g., FPN) for which feature maps need to match
      dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
      input size divisibility is required.
      """
    return 0

  def output_shape(self):
    """
        Returns:
            dict[str->ShapeSpec]
        """
    # this is a backward-compatible default
    return {
        name: ShapeSpec(channels=self._out_feature_channels[name],
                        stride=self._out_feature_strides[name])
        for name in self._out_features
    }
