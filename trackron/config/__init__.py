# Copyright (c) Facebook, Inc. and its affiliates.
from .compat import downgrade_config, upgrade_config
from .config import CfgNode, get_cfg, global_cfg, set_global_cfg, configurable
# from .model_configs import *
import trackron.config.model_configs as mc

__all__ = [
    "CfgNode",
    "get_cfg",
    "global_cfg",
    "set_global_cfg",
    "setup_cfg",
    "downgrade_config",
    "upgrade_config",
    "configurable",
]

def setup_cfg(args):
  """
    Create configs and perform basic setups.
    """
  cfg = get_cfg()
  config_func = getattr(mc, "add_{}_config".format(args.config_func), None)
  if config_func is None:
    raise NotImplemented('{} is not supported'.format(args.config_func))
  cfg = config_func(cfg)
  cfg.merge_from_file(args.config_file)
  cfg.merge_from_list(args.opts)
  cfg.freeze()
  return cfg