# Copyright (c) Facebook, Inc. and its affiliates.
# from trackron.layers import ShapeSpec
from fvcore.common.registry import Registry

# from .backbone import Backbone

DATA_PROCESSING_REGISTRY = Registry("DATA_PROCESSING")
DATA_PROCESSING_REGISTRY.__doc__ = """
Registry for Data processing, which process images and annotations for training
The registered object must be a callable that accepts two arguments:

1. A :class:`tracker.config.CfgNode`
2. B :str: training status

Registered object must return instance of :class:`Processing`.
"""


def build_processing_class(cfg, training=False):
  """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
  # if input_shape is None:
  #     input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

  process_class_name = cfg.PROCESSING_NAME
  return DATA_PROCESSING_REGISTRY.get(process_class_name)(cfg, training)
