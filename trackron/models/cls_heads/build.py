# Copyright (c) Facebook, Inc. and its affiliates.
# from trackron.layers import ShapeSpec
from fvcore.common.registry import Registry

# from .backbone import Backbone

CLS_HEAD_REGISTRY = Registry("CLS_HEAD")
CLS_HEAD_REGISTRY.__doc__ = """
Registry for LOCALIZATION HEAD, which output target localization based on consecutive images

The registered object must be a callable that accepts two arguments:

1. A :class:`tao.config.CfgNode`
2. A :class:`tracker.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_cls_head(cfg, input_shape=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    # if input_shape is None:
    #     input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    cls_head = CLS_HEAD_REGISTRY.get(cfg.NAME)(cfg)
    # assert isinstance(backbone, Backbone)
    return cls_head
