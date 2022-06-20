from trackron.config import CfgNode as CN
import copy

def maskhead_dynamic():
  cfg = CN()

  ### For Corner
  cfg.NAME = "DynamicMaskHead"
  cfg.INPUT_DIM = 256
  cfg.HIDDEN_DIM = 256
  cfg.UPSAMPLE = True
  cfg.POOL_SIZE = 14
  cfg.POOL_SCALES = [0.0625]
  cfg.POOL_SAMPLE_RATIO = 2
  cfg.SCALE_FACTOR = 2
  cfg.POOL_TYPE = "ROIAlignV2"
  return cfg
