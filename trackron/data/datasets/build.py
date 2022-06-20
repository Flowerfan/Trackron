from fvcore.common.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for DATASET, which calculate the loss between predictions and groundtruth
The registered object must be a callable that accepts two arguments:

1. A :class:`tracker.config.CfgNode`

Registered object must return instance of :class:`Backbone`.
"""





def no_processing(data):
  return data

def build_dataset(cfg, training):
  return DATASET_REGISTRY.get(cfg.CLASS_NAME)(cfg, training=training)

