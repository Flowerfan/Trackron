#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
trackron training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in trackron.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use trackron as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
from collections import OrderedDict
import torch
import traceback
from pathlib import Path

import trackron.utils.comm as comm
from trackron.checkpoint import TrackingCheckpointer, PeriodicCheckpointer
from trackron.data import build_tracking_loader, build_tracking_test_loader
from trackron.engine import default_argument_parser, default_setup, default_writers, launch
from trackron.models import build_model
from trackron.solvers import build_optimizer_scheduler
from trackron.utils.events import EventStorage
from trackron.config import setup_cfg

## for eval
from trackron.trackers import TrackingActor
from torch.nn.parallel import DistributedDataParallel as DDP
from trackron.evaluation import DETEvaluator, MOTEvaluator, SOTEvaluator, inference_on_dataset, print_csv_format

logger = logging.getLogger("trackron")


def get_evaluator(dataset_name, output_dir=None, tracking_mode='sot'):
  output_dir.mkdir(parents=True, exist_ok=True)
  if tracking_mode == 'sot':
    evaluator = SOTEvaluator(dataset_name, output_dir=output_dir)
  elif tracking_mode == 'mot':
    evaluator = [
        DETEvaluator(dataset_name, output_dir=output_dir),
        MOTEvaluator(dataset_name, output_dir=output_dir)
    ]
    # evaluator = [DETEvaluator(dataset_name, output_dir=output_dir)]

  else:
    raise NotImplementedError
  return evaluator


def write_test_metric(results, storage):
  if comm.is_main_process():
    save_dicts = {}
    for data_name, result in results.items():
      for metric_name, kvs in result.items():
        for k, v in kvs.items():
          name = f'{data_name}/{metric_name}/{k}'
          save_dicts[name] = v

    storage.put_scalars(smoothing_hint=False, **save_dicts)


def do_test(cfg, model, tracking_mode, debug_level=-1):
  model.eval()
  if isinstance(model, DDP):
    net = model.module
  else:
    net = model
  tracking_actor = TrackingActor(cfg,
                                 net,
                                 output_dir=cfg.OUTPUT_DIR,
                                 tracking_mode=tracking_mode,
                                 debug_level=debug_level,
                                 tracking_category=cfg.TRACKER.TRACKING_CATEGORY)
  results = OrderedDict()
  for idx, dataset_name in enumerate(cfg.DATASET.TEST.DATASET_NAMES):
    loader = build_tracking_test_loader(cfg, dataset_name)
    version = cfg.DATASET.TEST.VERSIONS[idx]
    if version is not "":
      dataset_name = f'{dataset_name}{version}'
    result_dir = Path(cfg.OUTPUT_DIR) / 'results' / dataset_name
    evaluator = get_evaluator(dataset_name,
                              output_dir=result_dir,
                              tracking_mode=tracking_mode)

    results_i = inference_on_dataset(tracking_actor, loader, evaluator)
    results[dataset_name] = results_i
    if comm.is_main_process() and tracking_mode == 'sot':
      logger.info(
          "Evaluation results for {} in csv format:".format(dataset_name))
      print_csv_format(results_i)
  return results


def do_val(val_iters, model, iteration=20):
  model.eval()
  metrics = {}
  with torch.no_grad():
    for i in range(iteration):
      _, _, metric = forward_step(val_iters, model)
      for k, v in metric.items():
        metrics[k] = metrics.get(k, 0.0) + v
  metric_reduced = {
      k: v.item() / iteration for k, v in comm.reduce_dict(metrics).items()
  }
  # if comm.is_main_process():
  #   storage.put_scalars(**metric_reduced)
  model.train()
  return metric_reduced


def forward_step(data_iters, model, max_fail=10, mode='sot'):
  fail_time = 0
  while fail_time < max_fail:
    try:
      data = next(data_iters)
      return model(data)
    except Exception:
      fail_time += 1
      traceback.print_exc()
      logger.warning('retry %d th time' % fail_time)
  traceback.print_exc()
  raise ValueError('Cannot get data')


def do_train(cfg, model, resume=False, tracking_mode='sot'):
  model.train()
  optimizer, scheduler = build_optimizer_scheduler(cfg, model)

  checkpointer = TrackingCheckpointer(model,
                                      cfg.OUTPUT_DIR,
                                      optimizer=optimizer,
                                      scheduler=scheduler)
  start_iter = (checkpointer.resume_or_load(
      cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
  max_iter = cfg.SOLVER.MAX_ITER

  periodic_checkpointer = PeriodicCheckpointer(checkpointer,
                                               cfg.SOLVER.CHECKPOINT_PERIOD,
                                               max_iter=max_iter)

  writers = default_writers(cfg.OUTPUT_DIR,
                            max_iter) if comm.is_main_process() else []

  # compared to "train_net.py", we do not support accurate timing and
  # precise BN here, because they are not trivial to implement in a small training loop
  train_loader = build_tracking_loader(cfg, training=True)
  val_loader = build_tracking_loader(cfg)
  logger.info("Starting training from iteration {}".format(start_iter))
  train_iters = iter(train_loader)
  with EventStorage(start_iter) as storage:
    # for data, iteration in zip(data_loader, range(start_iter, max_iter)):
    for iteration in range(start_iter, max_iter):
      storage.iter = iteration
      loss_dict, weighted_loss_dict, metrics = forward_step(train_iters,
                                                            model,
                                                            mode=tracking_mode)

      losses = sum(weighted_loss_dict.values())
      assert torch.isfinite(losses).all(), loss_dict

      loss_dict_reduced = {
          k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
      }
      metric_reduced = {
          k: v.item() for k, v in comm.reduce_dict(metrics).items()
      }
      losses_reduced = sum(loss for loss in loss_dict_reduced.values())
      if comm.is_main_process():
        storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        storage.put_scalars(**metric_reduced)

      optimizer.zero_grad()
      losses.backward()
      optimizer.step()
      storage.put_scalar("lr",
                         optimizer.param_groups[0]["lr"],
                         smoothing_hint=False)
      scheduler.step_update(iteration + 1)

      if (cfg.TEST.EVAL_PERIOD > 0 and
          (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and
          iteration != max_iter - 1):
        val_iters = iter(val_loader)
        metrics = do_val(val_iters, model)
        if comm.is_main_process():
          storage.put_scalars(**metrics, smoothing_hint=False)

      if (cfg.TEST.ETEST_PERIOD > 0 and
          (iteration + 1) % cfg.TEST.ETEST_PERIOD == 0 or
          iteration == max_iter - 1):
        results = do_test(cfg, model, tracking_mode=tracking_mode)
        write_test_metric(results, storage)
        # Compared to "train_net.py", the test results are not dumped to EventStorage
        model.train()
        comm.synchronize()

      if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or
                                         iteration == max_iter - 1):
        for writer in writers:
          writer.write()
      periodic_checkpointer.step(iteration)


def main(args):
  # cfg = setup(args)
  cfg = setup_cfg(args)
  default_setup(cfg, args)

  model = build_model(cfg)
  logger.info("Model:\n{}".format(model))
  if args.eval_only:
    TrackingCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume)
    return do_test(cfg, model, tracking_mode=args.mode, debug_level=args.debug)

  distributed = comm.get_world_size() > 1
  if distributed:
    model = DDP(model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True)

  do_train(cfg, model, resume=args.resume, tracking_mode=args.mode)
  # return do_test(cfg, model)


if __name__ == "__main__":
  args = default_argument_parser().parse_args()
  print("Command Line Args:", args)
  launch(
      main,
      args.num_gpus,
      num_machines=args.num_machines,
      machine_rank=args.machine_rank,
      dist_url=args.dist_url,
      args=(args,),
  )
