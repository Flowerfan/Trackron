from trackron.config import CfgNode as CN
import copy
from .boxhead_configs import boxhead_mlp, boxhead_target_trans
from .objective_configs import mot_default_objective
from .data_configs import default_dataloader_cfg, default_dataset_cfg

def mot_default():
  mot = CN()
  mot.OBJECTIVE = mot_default_objective()
  mot.DATASET = default_dataset_cfg()
  mot.DATALOADER = default_dataloader_cfg()
  return mot

def mot_transtrack():
  cfg = CN()
  cfg.NAME = "TransTrack"
  cfg.NUM_CLASS = 1
  cfg.FEATURE_LAYERS = ["layer2", "layer3", "layer4"]
  cfg.FEATURE_DIM = 256
  cfg.POSITION_EMBEDDING = "sine"

  ### MOT HEAD
  cfg.TWO_STAGE = False
  cfg.BOX_REFINE = True
  cfg.TRANSFORMER = CN()
  cfg.TRANSFORMER.HEADS = 8
  cfg.TRANSFORMER.ENC_LAYERS = 6
  cfg.TRANSFORMER.DEC_LAYERS = 6
  cfg.TRANSFORMER.DIM_FEEDFORWARD = 1024
  cfg.TRANSFORMER.NORM = 'relu'
  cfg.TRANSFORMER.DROPOUT = 0.1
  cfg.TRANSFORMER.NUM_FEATURE_LEVELS = 4
  cfg.TRANSFORMER.NUM_QUERIES = 500
  cfg.TRANSFORMER.ENC_POINTS = 4
  cfg.TRANSFORMER.DEC_POINTS = 4

  ### box head
  cfg.BOX_HEAD = boxhead_mlp()
  return cfg

def mot_dfdetr_proposal():
  cfg = CN()
  cfg.NAME = "TransTrack"
  cfg.NUM_CLASS = 1
  cfg.FEATURE_LAYERS = ["layer2", "layer3", "layer4"]
  cfg.NUM_FEATURE_LEVELS = 4
  cfg.FEATURE_DIM = 256
  cfg.POSITION_EMBEDDING = "sine"
  cfg.NUM_QUERIES = 500

  ### MOT HEAD
  cfg.TWO_STAGE = False
  cfg.BOX_REFINE = True

  ### ENCODER
  cfg.ENCODER = CN()
  cfg.ENCODER.HEADS = 8
  cfg.ENCODER.NUM_LAYERS = 6
  cfg.ENCODER.DIM_FEEDFORWARD = 1024
  cfg.ENCODER.NORM = 'relu'
  cfg.ENCODER.DROPOUT = 0.1
  cfg.ENCODER.NUM_POINTS = 4

  ### DECODER
  cfg.DECODER = CN()
  cfg.DECODER.HEADS = 8
  cfg.DECODER.NUM_LAYERS = 6
  cfg.DECODER.DIM_FEEDFORWARD = 1024
  cfg.DECODER.NORM = 'relu'
  cfg.DECODER.DROPOUT = 0.1
  cfg.DECODER.NUM_POINTS = 4

  ### box head
  cfg.BOX_HEAD = boxhead_mlp()

  cfg.TRACK_HEAD = boxhead_target_trans()
  return cfg
