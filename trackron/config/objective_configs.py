from trackron.config import CfgNode as CN
import copy

def mot_default_objective():
  ### MOT loss
  cfg = CN()
  cfg.NAME = 'MOTObjective'
  cfg.NUM_CLASS = 1230
  cfg.ITERATIONS = 6
  cfg.WEIGHT = CN()
  cfg.WEIGHT.LOSS_CE = 2.0
  cfg.WEIGHT.LOSS_BBOX = 5.0
  cfg.WEIGHT.LOSS_GIOU = 2.0
  cfg.WEIGHT.LOSS_MASK = 0.0
  cfg.WEIGHT.LOSS_CE_TRACK = 2.0
  cfg.WEIGHT.LOSS_BBOX_TRACK = 5.0
  cfg.WEIGHT.LOSS_GIOU_TRACK = 2.0
  cfg.WEIGHT.LOSS_CENTER_TRACK = 0.0
  return cfg

def sot_default_objective():
  # SOT LOSS
  cfg = CN()
  cfg.NAME = "SOTObjective"
  cfg.STACK = True
  cfg.WEIGHT = CN()
  cfg.WEIGHT.LOSS_CLS = 1.0
  cfg.WEIGHT.LOSS_BBOX = 5.0
  cfg.WEIGHT.LOSS_GIOU = 2.0
  cfg.WEIGHT.LOSS_MASK = 0.0
  return cfg

def sot_dimp_objective():
  cfg = sot_default_objective()
  cfg.NAME = 'DiMPObjective'
  cfg.LOSS_CLS = 'LBHinge'
  cfg.LOSS_CLS_THRESHOLD = 0.05
  cfg.LOSS_BBOX = 'IoUScore'
  cfg.WEIGHT.LOSS_CLS = 100.0
  cfg.WEIGHT.LOSS_BBOX = 1.0
  return cfg
