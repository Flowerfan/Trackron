from fvcore.common.registry import Registry

CLS_LOSS_REGISTRY = Registry('CLASSIFICATION_LOSS')
CLS_LOSS_REGISTRY.__doc__ = """
Registry for clssification loss.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

BOX_LOSS_REGISTRY = Registry('LOSS_BBOX')
BOX_LOSS_REGISTRY.__doc__ = """
Registry for Box loss.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""
