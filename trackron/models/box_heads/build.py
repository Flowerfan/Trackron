# Copyright (c) Facebook, Inc. and its affiliates.
# from trackron.layers import ShapeSpec
from fvcore.common.registry import Registry

# from .backbone import Backbone

BOX_HEAD_REGISTRY = Registry("BOX_HEAD")
BOX_HEAD_REGISTRY.__doc__ = """
Registry for Box Regression head, which extract box head
The registered object must be a callable that accepts two arguments:

1. A :class:`tracker.config.CfgNode`
2. A :class:`tracker.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_box_head(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    # if input_shape is None:
    #     input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    box_head_name = cfg.NAME
    box_head = BOX_HEAD_REGISTRY.get(box_head_name)(cfg)
    # assert isinstance(backbone, Backbone)
    return box_head
