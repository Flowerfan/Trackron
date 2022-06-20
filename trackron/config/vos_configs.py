from trackron.config import CfgNode as CN
import copy
from .objective_configs import sot_default_objective
from .data_configs import default_dataloader_cfg, default_dataset_cfg


def vos_default(cfg):
  vos = CN()
  vos.OBJECTIVE = sot_default_objective()
  vos.DATASET = default_dataset_cfg()
  vos.DATASET.TEMPLATE.SIZE = [480, 852]
  vos.DATASET.SEARCH.SIZE = [480, 852]
  vos.DATALOADER = default_dataloader_cfg()
  return vos
