from trackron.config import CfgNode as CN
import copy

from trackron.config.maskhead_configs import maskhead_dynamic
from .maskhead_configs import maskhead_dynamic

def boxhead_corner():
  cfg = CN()

  ### For Corner
  cfg.NAME = "Corner2"
  cfg.INPUT_DIM = 256
  cfg.HIDDEN_DIM = 256
  cfg.PATCH_DIM = 22 * 22
  cfg.OUTPUT_DIM = 4
  cfg.NUM_LAYERS = 3
  cfg.NORM = "BN"
  return cfg

def boxhead_center():
  cfg = CN()

  ### For Corner
  cfg.NAME = "Center"
  cfg.INPUT_DIM = 256
  cfg.HIDDEN_DIM = 256
  cfg.PATCH_DIM = 22 * 22
  cfg.DYNAMIC_DIM = 64
  cfg.NORM = "BN"
  return cfg

def boxhead_mlp():
  cfg = CN()
  cfg.NAME = "MLP"
  cfg.REFINE = True
  cfg.PATCH_DIM = 22 * 22
  cfg.INPUT_DIM = 256
  cfg.HIDDEN_DIM = 256
  cfg.OUTPUT_DIM = 4
  cfg.NUM_LAYERS = 3
  cfg.NORM = "BN"
  return cfg

def boxhead_rpn():
  cfg = CN()
  cfg.NAME = 'MultiRPN'
  cfg.ANCHOR_NUM = 5
  cfg.IN_CHANNELS = [256, 256, 256]
  cfg.WEIGHTED = True
  return cfg

def boxhead_target_trans():
  cfg = CN()
  cfg.NAME = "TargetTransformer"

  cfg.ITERATIONS = 6
  cfg.OUTPUT_SCORE = False
  cfg.FEATURE_DIM = 256
  cfg.DYNAMIC_DIM = 64
  cfg.NUM_DYNAMIC = 2
  cfg.DIM_FEEDFORWARD = 1024
  cfg.DROPOUT = 0.1
  cfg.POOL_SIZE = 7
  cfg.POOL_SCALES = [0.125, 0.0625, 0.03125, 0.015625]
  cfg.POOL_SAMPLE_RATIO = 2
  cfg.POOL_TYPE = "ROIAlignV2"
  cfg.NUM_BOX_LAYER = 3
  cfg.ACTIVATION = "relu"
  cfg.BOX_WEIGHTS = (2.0, 2.0, 1.0, 1.0)
  cfg.WITH_MASK = False
  cfg.MASK_HEAD = maskhead_dynamic()
  cfg.EXPAND_SCALE = 1.0
  return cfg


def boxhead_atom():
  cfg = CN()
  cfg.NAME = "AtomIoUNet"
  cfg.INPUT_DIM = [512, 1024]
  cfg.INTER_INPUT_DIM = [256, 256]
  cfg.INTER_OUT_DIM = [256, 256]
  cfg.INPUT_FEATURE_SCALE = [1 / 8, 1 / 16]
  return cfg
