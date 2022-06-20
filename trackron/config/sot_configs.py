import copy

from trackron.config import CfgNode as CN

from .boxhead_configs import (boxhead_atom, boxhead_corner, boxhead_mlp,
                              boxhead_target_trans, boxhead_rpn)
from .clshead_configs import clshead_dimp
from .objective_configs import sot_default_objective, sot_dimp_objective
from .data_configs import default_dataloader_cfg, default_dataset_cfg


def sot_default():
  cfg = CN()
  cfg.OBJECTIVE = sot_default_objective()
  cfg.DATASET = default_dataset_cfg()
  cfg.DATALOADER = default_dataloader_cfg()
  return cfg

def sot_dimp(cfg):
  cfg = CN()
  cfg.NAME = 'DIMP'
  cfg.FEATURE_STRIDE = 16
  cfg.CLS_HEAD = clshead_dimp()
  cfg.BOX_HEAD = boxhead_atom()
  return cfg

def sot_siamrpn(cfg):
  cfg = CN()
  cfg.NAME = 'SiamRPN'
  cfg.FEATURE_LAYER = ['layer2', 'layer3', 'layer4']

  cfg.NECK = CN()
  cfg.NECK.NAME = 'AdjustAllLayer'
  cfg.NECK.ENABLE = True
  cfg.NECK.IN_CHANNELS = [512, 1024, 2048]
  cfg.NECK.OUT_CHANNELS = [256, 256, 256]
  cfg.NECK.CENTER_SIZE = 7

  cfg.RPN_HEAD = boxhead_rpn()

  cfg.MASK_HEAD = CN()
  cfg.MASK_HEAD.ENABLE = False
  cfg.MASK_HEAD.REFINE = False
  return cfg

def sot_dfdetr():
  cfg = CN()
  cfg.NAME = "S3THead"
  cfg.FEATURE_LAYERS = ["layer2", "layer3", "layer4"]
  cfg.NUM_FEATURE_LAYERS = 4
  cfg.FEATURE_DIM = 256
  cfg.FEATURE_STRIDE = 16
  cfg.TWO_STAGE = False

  cfg.TARGET_TOKEN = False
  cfg.USE_QUERY_EMB = False

  ### ON EMBEDDING
  cfg.POSITION_EMBEDDING = 'sine'
  # ENCODER
  cfg.ENCODER = CN()
  cfg.ENCODER.NUM_LAYERS = 6
  cfg.ENCODER.NORM = 'relu'
  cfg.ENCODER.HEADS = 8
  cfg.ENCODER.DROPOUT = 0.1
  cfg.ENCODER.DIM_FEEDFORWARD = 1024

  # DECODER
  cfg.DECODER = CN()
  cfg.DECODER.NUM_LAYERS = 0
  cfg.DECODER.NORM = 'relu'
  cfg.DECODER.HEADS = 8
  cfg.DECODER.DROPOUT = 0.1
  cfg.DECODER.DIM_FEEDFORWARD = 1024
  cfg.DECODER.PRE_NORM = False

  ### Box Head
  cfg.BOX_HEAD = boxhead_corner()
  cfg.BOX_REFINE = False
  cfg.REFINE_HEAD = boxhead_target_trans()
  return cfg

def sot_token():
  """
    Add config for dfdetr
    """
  cfg = CN()
  cfg.NAME = "TOKENHead"
  cfg.FEATURE_LAYER = "layer3"
  cfg.FEATURE_DIM = 256
  cfg.FEATURE_STRIDE = 16
  cfg.KERNEL_SZ = 1

  cfg.USE_QUERY_EMB = True

  ### POSITION EMBEDDING
  cfg.POSITION_EMBEDDING = 'sine'

  # ENCODER
  cfg.ENCODER = CN()
  cfg.ENCODER.NUM_LAYERS = 6
  cfg.ENCODER.NORM = 'relu'
  cfg.ENCODER.HEADS = 8
  cfg.ENCODER.DROPOUT = 0.1
  cfg.ENCODER.DIM_FEEDFORWARD = 1024
  cfg.ENCODER.PRE_NORM = False

  # DECODER
  cfg.DECODER = CN()
  cfg.DECODER.NUM_LAYERS = 0
  cfg.DECODER.NORM = 'relu'
  cfg.DECODER.HEADS = 8
  cfg.DECODER.DROPOUT = 0.1
  cfg.DECODER.DIM_FEEDFORWARD = 1024
  cfg.DECODER.PRE_NORM = False
  # BOX HEAD
  cfg.BOX_HEAD = boxhead_corner()
  cfg.BOX_REFINE = False
  cfg.REFINE_HEAD = boxhead_target_trans()
  return cfg


def sot_s3t():
  """
    Saptial temporal transormer head.
	"""
  cfg = CN()
  cfg.NAME = "S3THead"
  cfg.FEATURE_LAYER = "layer3"
  cfg.FEATURE_DIM = 256
  cfg.FEATURE_STRIDE = 16
  cfg.KERNEL_SZ = 1

  cfg.TARGET_TOKEN = False
  cfg.USE_QUERY_EMB = False

  ### POSITION EMBEDDING
  cfg.POSITION_EMBEDDING = 'sine'

  # For Spatial Target Decoder
  cfg.TARGET_SPATIAL_DECODER = CN()
  cfg.TARGET_SPATIAL_DECODER.NUM_LAYERS = 6
  cfg.TARGET_SPATIAL_DECODER.NORM = 'relu'
  cfg.TARGET_SPATIAL_DECODER.HEADS = 8
  cfg.TARGET_SPATIAL_DECODER.DROPOUT = 0.1
  cfg.TARGET_SPATIAL_DECODER.DIM_FEEDFORWARD = 1024
  cfg.TARGET_SPATIAL_DECODER.PRE_NORM = False

  # For Temporal Target Encoder
  cfg.TARGET_TEMPORAL_ENCODER = CN()
  cfg.TARGET_TEMPORAL_ENCODER.NUM_LAYERS = 0
  cfg.TARGET_TEMPORAL_ENCODER.NORM = 'relu'
  cfg.TARGET_TEMPORAL_ENCODER.HEADS = 8
  cfg.TARGET_TEMPORAL_ENCODER.DROPOUT = 0.1
  cfg.TARGET_TEMPORAL_ENCODER.DIM_FEEDFORWARD = 1024
  cfg.TARGET_TEMPORAL_ENCODER.PRE_NORM = False

  # For Feature Spatial Temporal Encoder
  cfg.SPATIAL_TEMPORAL_ENCODER = CN()
  cfg.SPATIAL_TEMPORAL_ENCODER.NUM_LAYERS = 0
  cfg.SPATIAL_TEMPORAL_ENCODER.NORM = 'relu'
  cfg.SPATIAL_TEMPORAL_ENCODER.HEADS = 8
  cfg.SPATIAL_TEMPORAL_ENCODER.DROPOUT = 0.1
  cfg.SPATIAL_TEMPORAL_ENCODER.DIM_FEEDFORWARD = 1024
  cfg.SPATIAL_TEMPORAL_ENCODER.PRE_NORM = False

  # BOX HEAD
  cfg.BOX_HEAD = boxhead_corner()

  cfg.BOX_REFINE = False
  cfg.REFINE_HEAD = boxhead_target_trans()
  return cfg

def sot_decode():
  """
    Saptial temporal transormer head.
	"""
  cfg = CN()
  cfg.NAME = "S3THead"
  cfg.FEATURE_LAYER = "layer3"
  cfg.FEATURE_DIM = 256
  cfg.FEATURE_STRIDE = 16
  cfg.USE_QUERY_EMB = False
  cfg.KERNEL_SZ = 1

  ### POSITION EMBEDDING
  cfg.POSITION_EMBEDDING = 'sine'

  # For Spatial Target Decoder
  cfg.DECODER = CN()
  cfg.DECODER.NUM_LAYERS = 6
  cfg.DECODER.NORM = 'relu'
  cfg.DECODER.HEADS = 8
  cfg.DECODER.DROPOUT = 0.1
  cfg.DECODER.DIM_FEEDFORWARD = 1024
  cfg.DECODER.PRE_NORM = False

  # BOX HEAD
  cfg.BOX_HEAD = boxhead_corner()

  cfg.BOX_REFINE = False
  cfg.REFINE_HEAD = boxhead_target_trans()
  return cfg

def sot_stark():
  """Stark SOT head"""
  cfg = CN()
  cfg.NAME = 'STARK'
  cfg.FEATURE_LAYER = "layer3"
  cfg.FEATURE_DIM = 256
  cfg.FEATURE_STRIDE = 16

  cfg.POSITION_EMBEDDING = 'sine'
  ####Transformer
  cfg.ENCODER = CN()
  cfg.ENCODER.NUM_LAYERS = 6
  cfg.ENCODER.NORM = 'relu'
  cfg.ENCODER.HEADS = 8
  cfg.ENCODER.DROPOUT = 0.1
  cfg.ENCODER.DIM_FEEDFORWARD = 1024
  cfg.ENCODER.HIDDEN_DIM = 256
  cfg.ENCODER.PRE_NORM = False

  cfg.DECODER = CN()
  cfg.DECODER.NUM_LAYERS = 6
  cfg.DECODER.NORM = 'relu'
  cfg.DECODER.HEADS = 8
  cfg.DECODER.DROPOUT = 0.1
  cfg.DECODER.DIM_FEEDFORWARD = 1024
  cfg.DECODER.HIDDEN_DIM = 256
  cfg.DECODER.PRE_NORM = False

  # BOX HEAD
  cfg.BOX_HEAD = boxhead_corner()

  # LOSS
  cfg.GIOU_WEIGHT = 2.0
  cfg.L1_WEIGHT = 5.0
  cfg.NO_OBJECT_WEIGHT = 0.1
  return cfg
