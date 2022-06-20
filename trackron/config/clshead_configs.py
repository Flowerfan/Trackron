from trackron.config import CfgNode as CN
import copy

def clshead_dimp():
  cfg = CN()
  cfg.NAME = 'DIMP_CLS_HEAD'

  cfg.FILTER = CN()
  cfg.FILTER.NAME = "FilterInitializerLinear"
  cfg.FILTER.FILTER_SIZE = 4
  cfg.FILTER.FEATURE_STRIDE = 16
  cfg.FILTER.FEATURE_DIM = 512
  cfg.FILTER.POOL_SQUARE = False
  cfg.FILTER.NORM = False

  cfg.OPTIMIZER = CN()
  cfg.OPTIMIZER.NAME = "DiMPSteepestDescentGN"
  cfg.OPTIMIZER.ITER = 6
  cfg.OPTIMIZER.STEP = 1.0
  cfg.OPTIMIZER.GAUSS_SIGMA = 1.0
  cfg.OPTIMIZER.NUM_DIST_BINS = 5
  cfg.OPTIMIZER.BIN_DISPLACEMENT = 1.0
  cfg.OPTIMIZER.MASK_FACOTR = 4.0
  cfg.OPTIMIZER.INIT_FILTER_REG = 1e-2
  cfg.OPTIMIZER.MIN_FILTER_REG = 1e-3
  cfg.OPTIMIZER.SCORE_ACT = 'relu'
  cfg.OPTIMIZER.MASK_ACT = 'sigmoid'

  # DIMP Feature extraction
  cfg.EXTRACTOR = CN()
  cfg.EXTRACTOR.NAME = "residual_bottleneck"
  cfg.EXTRACTOR.FEATURE_DIM = 256
  cfg.EXTRACTOR.NUM_BLOCKS = 0
  cfg.EXTRACTOR.FEATURE_NORM = False
  cfg.EXTRACTOR.FINAL_CONV = False
  cfg.EXTRACTOR.FILTER_SIZE = cfg.FILTER.FILTER_SIZE
  cfg.EXTRACTOR.OUT_DIM = 256
  cfg.EXTRACTOR.INTERP_CAT = False
  cfg.EXTRACTOR.ACTIVATION = False
  cfg.EXTRACTOR.POOL = False
  return cfg
