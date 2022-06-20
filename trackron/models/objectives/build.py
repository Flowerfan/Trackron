# Copyright (c) Facebook, Inc. and its affiliates.
# from trackron.layers import ShapeSpec
from fvcore.common.registry import Registry

# from .backbone import Backbone

OBJECTIVE_REGISTRY = Registry("OBJECTIVE")
OBJECTIVE_REGISTRY.__doc__ = """
Registry for Objective, which calculate the loss between predictions and groundtruth
The registered object must be a callable that accepts two arguments:

1. A :class:`tracker.config.CfgNode`

Registered object must return instance of :class:`Backbone`.
"""


def build_objective(cfg):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    # if input_shape is None:
    #     input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    objective_name = cfg.NAME
    return  OBJECTIVE_REGISTRY.get(objective_name)(cfg)
