from fvcore.common.registry import Registry

EXTRACTOR_REGISTRY = Registry("EXTRACTOR")
EXTRACTOR_REGISTRY.__doc__ = """
Registry for Box Regression head, which extract box head
The registered object must be a callable that accepts two arguments:

1. A :class:`tracker.config.CfgNode`
2. A :class:`tracker.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""



def build_extractor(cfg):
  return EXTRACTOR_REGISTRY.get(cfg.NAME)(cfg)
