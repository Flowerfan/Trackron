# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging
import pickle
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel
from typing import Any

import trackron.utils.comm as comm
from trackron.utils.env import TORCH_VERSION

from .model_loading import align_and_update_state_dicts


class TrackingCheckpointer(Checkpointer):
  """
    Same as :class:`Checkpointer`, but is able to handle models in Tracking
    model zoo, and apply conversions for legacy models.
    """

  def __init__(self,
               model,
               save_dir="",
               *,
               save_to_disk=None,
               **checkpointables):
    is_main_process = comm.is_main_process()
    if isinstance(model, DistributedDataParallel):
      self.dist_model = model
    super().__init__(
        model,
        save_dir,
        save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
        **checkpointables,
    )

  def save(self, name: str, **kwargs: Any) -> None:
      """
      only save model parameters for final model

      Args:
          name (str): name of the file.
          kwargs (dict): extra arbitrary data to save.
      """
      if not self.save_dir or not self.save_to_disk:
          return
      if name != 'model_final':
        super().save(name, **kwargs)

      basename = "{}.pth".format(name)
      save_file = os.path.join(self.save_dir, basename)
      assert os.path.basename(save_file) == basename, basename

      data = {}
      data["model"] = self.model.state_dict()
      self.logger.info("Saving checkpoint to {}".format(save_file))
      with self.path_manager.open(save_file, "wb") as f:
          torch.save(data, f)
      self.tag_last_checkpoint(basename)

  def load(self, path, *args, **kwargs):
    need_sync = False

    if path and comm.get_world_size() > 1:
      logger = logging.getLogger(__name__)
      path = self.path_manager.get_local_path(path)
      has_file = os.path.isfile(path)
      all_has_file = comm.all_gather(has_file)
      if not all_has_file[0]:
        raise OSError(f"File {path} not found on main worker.")
      if not all(all_has_file):
        logger.warning(f"Not all workers can read checkpoint {path}. "
                       "Training may fail to fully resume.")
        # TODO: broadcast the checkpoint file contents from main
        # worker, and load from it instead.
        need_sync = True
      if not has_file:
        path = None  # don't load if not readable
    ret = super().load(path, *args, **kwargs)

    if need_sync:
      logger.info("Broadcasting model states from main worker ...")
      if TORCH_VERSION >= (1, 7):
        self.dist_model._sync_params_and_buffers()
    return ret

  def _load_file(self, filename):
    if filename.endswith(".pkl"):
      with open(filename, "rb") as f:
        data = pickle.load(f, encoding="latin1")
      if "model" in data and "__author__" in data:
        # file is in Tracker model zoo format
        self.logger.info("Reading a file from '{}'".format(data["__author__"]))
        return data
      else:
        raise ValueError('wrong model file')

    loaded = super()._load_file(filename)  # load native pth checkpoint
    if "model" not in loaded:
      loaded = {"model": loaded}
    if ".torch/iopath_cache" in filename:
      loaded["matching_heuristics"] = True
    return loaded

  def _load_model(self, checkpoint):
    if checkpoint.get("matching_heuristics", False):
      self._convert_ndarray_to_tensor(checkpoint["model"])
      # convert weights by name-matching heuristics
      checkpoint["model"] = align_and_update_state_dicts(
          self.model.state_dict(),
          checkpoint["model"],
      )
    incompatible = super()._load_model(checkpoint)

    model_buffers = dict(self.model.named_buffers(recurse=False))
    for k in ["pixel_mean", "pixel_std"]:
      # Ignore missing key message about pixel_mean/std.
      # Though they may be missing in old checkpoints, they will be correctly
      # initialized from config anyway.
      if k in model_buffers:
        try:
          incompatible.missing_keys.remove(k)
        except ValueError:
          pass
    return incompatible
