from trackron.config import CfgNode as CN

from trackron.config.vos_configs import vos_default
from .objective_configs import sot_default_objective, mot_default_objective, sot_dimp_objective
from .sot_configs import *
from .mot_configs import *
from .vos_configs import vos_default
from .boxhead_configs import boxhead_corner, boxhead_mlp, boxhead_target_trans, boxhead_center



def add_sot_config(cfg):
    cfg.MODEL.SOT = sot_s3t()
    cfg.SOT = sot_default()
    return cfg

def add_dimp_config(cfg):
    cfg.MODEL.SOT = sot_dimp(cfg)
    cfg.SOT = sot_default()
    cfg.SOT.OBJECTIVE = sot_dimp_objective()
    cfg.SOT.DATASET.FEATURE_STRIDE = cfg.MODEL.SOT.FEATURE_STRIDE
    cfg.SOT.DATASET.FILTER_SIZE = cfg.MODEL.SOT.CLS_HEAD.FILTER.FILTER_SIZE
    cfg.TRACKER.MEMORY_SIZE = 50
    cfg.TRACKER.PARAMETER = 'dimp50'
    return cfg

def add_siamrpn_config(cfg):
    cfg.MODEL.SOT = sot_siamrpn(cfg)
    cfg.SOT = sot_default()
    cfg.SOT.OBJECTIVE = sot_default_objective()
    cfg.TRACKER.PARAMETER = 'siamrpn'
    return cfg


def add_stark_config(cfg):
    """
    Add config for Stark.
    """
    cfg = add_sot_config(cfg)
    cfg.MODEL.SOT = sot_stark()
    return cfg

def add_fairmot_config(cfg):
    """
    config for FairMOT
    """

    cfg.MOT = mot_default()
    cfg.MODEL.NUM_CLASS = 1
    cfg.MODEL.REID_DIM = 128
    cfg.MODEL.HEAD_CONV = 256
    cfg.MODEL.REG_OFFSET = True
    cfg.MODEL.LTRB = True
    cfg.MODEL.MAX_OBJECTS = 500
    return cfg

def add_transtrack_config(cfg):
    """config for TransTrack
  """
    cfg.MODEL.MOT = mot_transtrack()
    cfg.MOT = mot_default()
    return cfg

def add_bytetrack_config(cfg):
    """config for Byterack
  """
    cfg.MODEL.NUM_CLASS = 1
    cfg.MOT = mot_default()
    return cfg

def add_sot_s3t_config(cfg):
    cfg.MODEL.SOT = sot_s3t()
    cfg.SOT = sot_default()
    return cfg

def add_sot_dfdetr_config(cfg):
    cfg = add_sot_s3t_config(cfg)
    cfg.MODEL.SOT = sot_dfdetr()
    return cfg

def add_sot_token_config(cfg):
    cfg = add_sot_s3t_config(cfg)
    cfg.MODEL.SOT = sot_token()
    return cfg

def add_sot_decode_config(cfg):
    cfg = add_sot_s3t_config(cfg)
    cfg.MODEL.SOT = sot_decode()
    return cfg

def add_siamese_ut_decode_proposal_config(cfg):
    ### Tracking Heads
    cfg.MODEL.SOT = sot_decode()
    cfg.MODEL.MOT = mot_dfdetr_proposal()

    cfg.SOT = sot_default()
    cfg.MOT = mot_default()

    return cfg


def add_siamese_ut_token_config(cfg):
    """
    Add config for dfdetr
    """
    ### Tracking Heads
    cfg.MODEL.SOT = sot_token()
    cfg.MODEL.MOT = mot_transtrack()

    cfg.SOT = sot_default()
    cfg.MOT = mot_default()

    return cfg

def add_siamese_ut_token_center_config(cfg):
    """
    Add config for dfdetr
    """
    ### Tracking Heads
    cfg = add_siamese_ut_token_config(cfg)
    cfg.MODEL.SOT.BOX_HEAD = boxhead_center()

    return cfg


def add_siamese_ot_config(cfg):
    """
    Add config for SQUENCE S3T.
    """

    ### model part
    cfg.MODEL.NUM_CLASS = 1
    cfg.MODEL.FEATURE_LAYERS = ["layer2", "layer3", "layer4"]
    cfg.MODEL.NUM_FEATURE_LAYERS = 4
    cfg.MODEL.FEATURE_DIM = 256
    cfg.MODEL.POSITION_EMBEDDING = "sine"

    ### MOT HEAD
    cfg.MODEL.TWO_STAGE = False
    cfg.MODEL.BOX_REFINE = True
    cfg.MODEL.TRANSFORMER = CN()
    cfg.MODEL.TRANSFORMER.HEADS = 8
    cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
    cfg.MODEL.TRANSFORMER.DEC_LAYERS = 6
    cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 1024
    cfg.MODEL.TRANSFORMER.NORM = 'relu'
    cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
    cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS = 4
    cfg.MODEL.TRANSFORMER.NUM_QUERIES = 500
    cfg.MODEL.TRANSFORMER.ENC_POINTS = 4
    cfg.MODEL.TRANSFORMER.DEC_POINTS = 4

    cfg.MODEL.BOX_HEAD = boxhead_mlp()

    cfg.MODEL.SOT = CN()
    cfg.MODEL.SOT.BOX_HEAD = boxhead_corner()

    cfg.SOT = sot_default()
    cfg.MOT = mot_default()

    return cfg

def add_utt_config(cfg):
    """
    Add config for utt
    """

    ### model part
    cfg.MODEL.NUM_CLASS = 1
    cfg.MODEL.FEATURE_LAYERS = ["layer2", "layer3", "layer4"]
    cfg.MODEL.NUM_FEATURE_LAYERS = 4
    cfg.MODEL.FEATURE_DIM = 256
    cfg.MODEL.OBJECT_SIZE = 1
    cfg.MODEL.POSITION_EMBEDDING = "sine"
    cfg.MODEL.NUM_QUERIES = 500
    cfg.MODEL.TWO_STAGE = False
    cfg.MODEL.BOX_REFINE = True
    cfg.MODEL.NORM = 'BN'


    # cfg.MODEL.USE_QUERY_EMB = False
    # ENCODER
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.NUM_LAYERS = 6
    cfg.MODEL.ENCODER.NORM = 'relu'
    cfg.MODEL.ENCODER.HEADS = 8
    cfg.MODEL.ENCODER.DROPOUT = 0.1
    cfg.MODEL.ENCODER.DIM_FEEDFORWARD = 1024
    cfg.MODEL.ENCODER.NUM_POINTS = 4

    # MOT DETECTION DECODER
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.NUM_LAYERS = 6
    cfg.MODEL.DECODER.NORM = 'relu'
    cfg.MODEL.DECODER.HEADS = 8
    cfg.MODEL.DECODER.DROPOUT = 0.1
    cfg.MODEL.DECODER.DIM_FEEDFORWARD = 1024
    cfg.MODEL.DECODER.PRE_NORM = False
    cfg.MODEL.DECODER.NUM_POINTS = 4



    ### Detection
    cfg.MODEL.BOX_HEAD = boxhead_mlp()

    ### Tracking Head
    cfg.MODEL.TRACK_HEAD = boxhead_target_trans()

    ### SOT
    cfg.MODEL.SOT = sot_decode()
    cfg.MODEL.SOT.POOL_SIZE = 7
    cfg.MODEL.SOT.POOL_SCALES = [0.0625]
    cfg.MODEL.SOT.POOL_SAMPLE_RATIO = 2
    cfg.MODEL.SOT.POOL_TYPE = "ROIAlignV2"

    #### MOT
    cfg.MODEL.MOT = CN()
    cfg.MODEL.MOT.POOL_SIZE = 7
    cfg.MODEL.MOT.POOL_SCALES = [0.125, 0.0625, 0.03125, 0.015625]
    cfg.MODEL.MOT.POOL_SAMPLE_RATIO = 2
    cfg.MODEL.MOT.POOL_TYPE = "ROIAlignV2"

    #### MOTS
    cfg.MODEL.MOT.USE_SEGMENTATION = False
    cfg.MODEL.MOT.SEGMENTATION = boxhead_mlp()


    cfg.SOT = sot_default()
    cfg.MOT = mot_default()
    cfg.VOS = vos_default(cfg)

    return cfg
