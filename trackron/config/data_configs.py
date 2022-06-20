from trackron.config import CfgNode as CN


def default_dataset_cfg():
    cfg = CN()
    cfg.ROOT = './data'
    cfg.CLASS_NAME = "SequenceDataset"
    cfg.PROCESSING_NAME = "SequenceProcessing"
    cfg.BOX_MODE = 'xywh'
    cfg.CROP_TYPE = 'inside_major'
    cfg.SAMPLE_MODE = "casual"  # sampling methods
    cfg.MAX_SAMPLE_INTERVAL = 200
    cfg.LABLE_SIGMA = 0.05

    # DATA.TRAIN
    cfg.TRAIN = CN()
    cfg.TRAIN.DATASET_NAMES = ["GOT10K_vottrain"]
    cfg.TRAIN.MODE = 'sequence' ### pair or sequence
    cfg.TRAIN.DATASETS_RATIO = [1]
    cfg.TRAIN.SAMPLE_PER_EPOCH = 1000000
    cfg.TRAIN.FRAMES = 1
    ### mot
    cfg.TRAIN.MIN_SIZE = [480]
    cfg.TRAIN.MIN_SIZE_SAMPLING= "choice"
    cfg.TRAIN.MAX_SIZE = 1333

    cfg.TRAIN.PROPOSALS = CN()
    cfg.TRAIN.PROPOSALS.MIN_IOU = 0.1
    cfg.TRAIN.PROPOSALS.BOXES_PER_TARGET = 8
    cfg.TRAIN.PROPOSALS.SIGMA_FACTOR = [0.01, 0.05, 0.1, 0.2, 0.3]

    # DATA.VAL
    cfg.VAL = CN()
    cfg.VAL.DATASET_NAMES = ["GOT10K_votval"]
    cfg.VAL.FRAMES = 10
    cfg.VAL.DATASETS_RATIO = [1]
    cfg.VAL.SAMPLE_PER_EPOCH = 10000

    # DATA.TEST
    cfg.TEST = CN()
    cfg.TEST.DATASET_NAMES = []
    cfg.TEST.VERSIONS = []
    cfg.TEST.SPLITS = []

    # DATA.SEARCH
    cfg.SEARCH = CN()
    cfg.SEARCH.FRAMES = 1  # number of search frames for multiple frames training
    cfg.SEARCH.SIZE = 320
    cfg.SEARCH.FACTOR = 5.0
    cfg.SEARCH.CENTER_JITTER = 4.5
    cfg.SEARCH.SCALE_JITTER = 0.5
    cfg.SEARCH.OUT_SIZE = 40


    # DATA.TEMPLATE
    cfg.TEMPLATE = CN()
    cfg.TEMPLATE.FRAMES = 1
    cfg.TEMPLATE.SIZE = 320
    cfg.TEMPLATE.FACTOR = 2.0
    cfg.TEMPLATE.CENTER_JITTER = 0.0
    cfg.TEMPLATE.SCALE_JITTER = 0.0
    cfg.TEMPLATE.OUT_SIZE = 7


    ### DATA.ANCHOR SiamRPN
    cfg.ANCHOR = CN()
    cfg.ANCHOR.STRIDE = 8
    cfg.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]
    cfg.ANCHOR.SCALES = [8]
    cfg.ANCHOR.ANCHOR_NUM = 5
    return cfg


def default_dataloader_cfg():
    cfg = CN()
    # Number of data loading threads
    cfg.NUM_WORKERS = 0
    # If True, each batch should contain only images for which the aspect ratio
    # is compatible. This groups portrait images together, and landscape images
    # are not batched with portrait images.
    cfg.ASPECT_RATIO_GROUPING = True
    # Options: TrainingSampler, RepeatFactorTrainingSampler
    cfg.SAMPLER_TRAIN = "TrainingSampler"
    # Repeat threshold for RepeatFactorTrainingSampler
    cfg.REPEAT_THRESHOLD = 0.0
    # Tf True, when working on datasets that have instance annotations, the
    # training dataloader will filter out images without associated annotations
    cfg.FILTER_EMPTY_ANNOTATIONS = True

    cfg.STACK_DIM = 0
    cfg.COLLATE_FN = None

    # Number of images per batch across all machines. This is also the number
    # of training images per step (i.e. per iteration). If we use 16 GPUs
    # and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
    # May be adjusted automatically if REFERENCE_WORLD_SIZE is set.
    cfg.BATCH_SIZE = 16
    return cfg
