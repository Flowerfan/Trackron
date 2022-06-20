# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'DiMPNet'
_C.MODEL.DEVICE = "cuda"

# Path (a file path, or URL like tracker://.., https://..) to a checkpoint file
# to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

_C.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.MODEL.PIXEL_STD = [0.229, 0.224, 0.225]
_C.MODEL.POSITION_EMBEDDING = 'sine'
_C.MODEL.HIDDEN_DIM = 512

# MODEL.BACKBONE
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "resnet50"  # resnet50, resnext101_32x8d
_C.MODEL.BACKBONE.PRETRAIN = False
_C.MODEL.BACKBONE.OUTPUT_LAYERS = ["layer2", "layer3", "layer4"]
_C.MODEL.BACKBONE.CLS_LAYERS = ["layer3"]
_C.MODEL.BACKBONE.STRIDE = 16
_C.MODEL.BACKBONE.DILATION = False
_C.MODEL.BACKBONE.NORM = "BN"
_C.MODEL.BACKBONE.USE_POSITION = False
_C.MODEL.BACKBONE.FROZEN_STAGES = -1

### OBJECTIVE for calculating model loss
_C.OBJECTIVE = CN()
_C.OBJECTIVE.NAME = 'DiMPObjective'

#### TRAINER for training model
_C.TRAINER = CN()

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# See tracker/solver/build.py for OPTIMIZER and LR scheduler options
_C.SOLVER.OPTIMIZER_NAME = "adam"

_C.SOLVER.MAX_ITER = 60000

####OPTIMIZER
_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BASE_LR_BACKBONE = 1.0

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 0.01
_C.SOLVER.BETAS = (0.9, 0.98)
_C.SOLVER.EPS = 1e-8
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000

# The reference number of workers (GPUs) this config is meant to train with.
# It takes no effect when set to 0.
# With a non-zero value, it will be used by DefaultTrainer to compute a desired
# per-worker batch size, and then scale the other related configs (total batch size,
# learning rate, etc) to match the per-worker batch size.
# See documentation of `DefaultTrainer.auto_scale_workers` for details:
_C.SOLVER.REFERENCE_WORLD_SIZE = 0

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Enable automatic mixed precision for training
# Note that this does not change model's inference behavior.
# To use AMP in inference, run inference under autocast()
_C.SOLVER.AMP = CN({"ENABLED": False})

#### FOR LR_SCHEDULER
_C.SOLVER.LR_SCHEDULER = CN()
_C.SOLVER.LR_SCHEDULER.NAME = "step"
_C.SOLVER.LR_SCHEDULER.LR_NOISE = None
_C.SOLVER.LR_SCHEDULER.LR_MIN = 0.
_C.SOLVER.LR_SCHEDULER.LR_CYCLE_MUL = 1.
_C.SOLVER.LR_SCHEDULER.LR_CYCLE_LIMIT = 1
_C.SOLVER.LR_SCHEDULER.LR_NOISE_PCT = 0.67
_C.SOLVER.LR_SCHEDULER.LR_NOISE_STD = 1.
_C.SOLVER.LR_SCHEDULER.LR_PATIENCE_ITERS = 0
_C.SOLVER.LR_SCHEDULER.DECAY_RATE = 0.1
_C.SOLVER.LR_SCHEDULER.DECAY_STEP = 30
_C.SOLVER.LR_SCHEDULER.WARMUP_ITERS = 0
_C.SOLVER.LR_SCHEDULER.WARMUP_LR = 0.000001
_C.SOLVER.LR_SCHEDULER.COOL_DOWN = 0

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0
_C.TEST.ETEST_PERIOD = 0
# The sigmas used to calculate keypoint OKS. See http://cocodataset.org/#keypoints-eval
# When empty, it will use the defaults in COCO.
# Otherwise it should be a list[float] with the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
_C.TEST.KEYPOINT_OKS_SIGMAS = []
# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
_C.TEST.OBJECTS_PER_VIDEO = 100

_C.TEST.AUG = CN({"ENABLED": False})
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
_C.TEST.AUG.MAX_SIZE = 4000
_C.TEST.AUG.FLIP = True

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200

# ---------------------------------------------------------------------------- #
# Specific tracking params
# ---------------------------------------------------------------------------- #
_C.TRACKER = CN()
_C.TRACKER.NAME = ""
_C.TRACKER.TRACKING_CATEGORY = None
_C.TRACKER.MEMORY_SIZE = 1
_C.TRACKER.OUTPUT_SCORE = False
_C.TRACKER.DEBUG_LEVEL = 0
_C.TRACKER.VISUALIZATION = False
_C.TRACKER.USE_MOTION = False
_C.TRACKER.USE_KALMAN = False
_C.TRACKER.PUBLIC_DETECTION = False
_C.TRACKER.DETECTION_THRESH = 0.4
_C.TRACKER.NMS_THRESH = 0.4
_C.TRACKER.TRACKING_THRESH = 0.4
_C.TRACKER.MATCHING_THRESH = 1.2


### wether to use sync bn
_C.SYNC_BN = False
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed increases
# reproducibility but does not guarantee fully deterministic behavior.
# Disabling all parallelism further increases reproducibility.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable.
_C.VIS_PERIOD = 0

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from trackron.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()