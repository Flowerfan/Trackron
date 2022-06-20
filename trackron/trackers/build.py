from fvcore.common.registry import Registry
from torch.nn.parallel import DistributedDataParallel 


TRACKER_REGISTRY = Registry("TRACKER")
TRACKER_REGISTRY.__doc__ = """
Registry for Tracking Module, which tracking the video sequence for evaluation
The registered object must be a callable that accepts two arguments:

1. A :class:`tracker.config.CfgNode`

Registered object must return instance of :class:`Backbone`.
"""



def build_tracker(cfg, net, tracking_mode='sot'):
  """[summary]

  Args:
      cfg ([type]): [description]
  TODO
  """
  tracker_name = cfg.TRACKER.NAME
  if isinstance(net, DistributedDataParallel):
    net = net.module
  tracker = TRACKER_REGISTRY.get(tracker_name)(cfg, net, tracking_mode)
  return tracker 