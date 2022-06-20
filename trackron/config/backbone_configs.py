import copy

from trackron.config import CfgNode as CN


def yolox_cfg(cfg):
  bbone_cfg = cfg.BACKBONE
  bbone_cfg.DEPTH = 1.0
  bbone_cfg.WIDTH = 1.0
  return bbone_cfg
