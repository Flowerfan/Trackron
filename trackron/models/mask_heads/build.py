# Copyright (c) Facebook, Inc. and its affiliates.
# from trackron.layers import ShapeSpec
from fvcore.common.registry import Registry

# from .backbone import Backbone

MASK_HEAD_REGISTRY = Registry("MASK_HEAD")
MASK_HEAD_REGISTRY.__doc__ = """
Registry for MASK head, which extract box head
The registered object must be a callable that accepts two arguments:

1. A :class:`tracker.config.CfgNode`
2. A :class:`tracker.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_mask_head(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    # if input_shape is None:
    #     input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    mask_head_name = cfg.NAME
    mask_head = MASK_HEAD_REGISTRY.get(mask_head_name)(cfg)
    return mask_head
