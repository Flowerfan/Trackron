import os
import copy
import torch
import logging
import itertools
import contextlib
import numpy as np
import seaborn as sns
from PIL import Image
from collections import OrderedDict
from pathlib import Path
from .evaluator import DatasetEvaluator
from trackron.utils import comm
from trackron.config import CfgNode

_PALETTE = (np.array(sns.color_palette(n_colors=256)) * 255).astype('uint8').ravel()



def calc_err_center(pred_bb, anno_bb, normalized=False):
  pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
  anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

  if normalized:
    pred_center = pred_center / anno_bb[:, 2:]
    anno_center = anno_center / anno_bb[:, 2:]

  err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
  return err_center


def calc_iou_overlap(pred_bb, anno_bb):
  tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
  br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0,
                 anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
  sz = (br - tl + 1.0).clamp(0)

  # Area
  intersection = sz.prod(dim=1)
  union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

  return intersection / union


def calc_seq_err_robust(pred_bb, anno_bb, dataset="otb", target_visible=None):
  pred_bb = pred_bb.clone()

  # Check if invalid values are present
  if torch.isnan(pred_bb).any() or (pred_bb[:, 2:] < 0.0).any():
    raise Exception('Error: Invalid results')

  if torch.isnan(anno_bb).any():
    if dataset == 'uav':
      pass
    else:
      raise Exception('Warning: NaNs in annotation')

  if (pred_bb[:, 2:] == 0.0).any():
    for i in range(1, pred_bb.shape[0]):
      if (pred_bb[i, 2:] == 0.0).any() and not torch.isnan(anno_bb[i, :]).any():
        pred_bb[i, :] = pred_bb[i - 1, :]

  if pred_bb.shape[0] != anno_bb.shape[0]:
    if dataset == 'lasot':
      if pred_bb.shape[0] > anno_bb.shape[0]:
        # For monkey-17, there is a mismatch for some trackers.
        pred_bb = pred_bb[:anno_bb.shape[0], :]
      else:
        raise Exception('Mis-match in tracker prediction and GT lengths')
    else:
      # print('Warning: Mis-match in tracker prediction and GT lengths')
      if pred_bb.shape[0] > anno_bb.shape[0]:
        pred_bb = pred_bb[:anno_bb.shape[0], :]
      else:
        pad = torch.zeros(
            (anno_bb.shape[0] - pred_bb.shape[0], 4)).type_as(pred_bb)
        pred_bb = torch.cat((pred_bb, pad), dim=0)

  pred_bb[0, :] = anno_bb[0, :]

  if target_visible is not None:
    target_visible = torch.tensor(target_visible, dtype=torch.bool)
    valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
  else:
    valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

  err_center = calc_err_center(pred_bb, anno_bb)
  err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
  err_overlap = calc_iou_overlap(pred_bb, anno_bb)

  # handle invalid anno cases
  if dataset in ['uav']:
    err_center[~valid] = -1.0
  else:
    err_center[~valid] = float("Inf")
  err_center_normalized[~valid] = -1.0
  err_overlap[~valid] = -1.0

  if dataset == 'lasot':
    err_center_normalized[~target_visible] = float("Inf")
    err_center[~target_visible] = float("Inf")

  if torch.isnan(err_overlap).any():
    raise Exception('Nans in calculated overlap')
  return err_overlap, err_center, err_center_normalized, valid


def save_tracker_output(seq_name, out_dir: Path, output: dict):
  """Saves the output of the tracker."""
  base_results_path = out_dir / seq_name

  def save_bb(file, data):
    tracked_bb = np.array(data).astype(float)
    np.savetxt(file, tracked_bb, delimiter='\t', fmt='%1.2f')
    # tracked_bb = np.array(data).astype(int)
    # np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

  def save_time(file, data):
    exec_times = np.array(data).astype(float)
    np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

  def _convert_dict(input_dict):
    data_dict = {}
    for elem in input_dict:
      for k, v in elem.items():
        if k in data_dict.keys():
          data_dict[k].append(v)
        else:
          data_dict[k] = [
              v,
          ]
    return data_dict

  for key, data in output.items():
    # If data is empty
    if not data:
      continue
    if key == 'target_bbox':
      if isinstance(data[0], (dict, OrderedDict)):
        data_dict = _convert_dict(data)

        for obj_id, d in data_dict.items():
          bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
          save_bb(bbox_file, d)
      else:
        # Single-object mode
        bbox_file = '{}.txt'.format(base_results_path)
        save_bb(bbox_file, data)
    elif key == 'time':
      if isinstance(data[0], dict):
        data_dict = _convert_dict(data)

        for obj_id, d in data_dict.items():
          timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
          save_time(timings_file, d)
      else:
        timings_file = '{}_time.txt'.format(base_results_path)
        save_time(timings_file, data)
    elif key == 'segmentation':
      base_results_path.mkdir(exist_ok=True)
      for idx, mask in enumerate(output['segmentation']):
        png_path = base_results_path / '{:05d}.png'.format(idx)
        img = Image.fromarray(mask)
        img.putpalette(_PALETTE)
        img.save(png_path, format='PNG')


_EVAL_SETS = ['otb', 'lasot']


class SOTEvaluator(DatasetEvaluator):

  def __init__(self,
               dataset_name,
               distributed=True,
               output_dir=None,
               tasks=None):
    self._logger = logging.getLogger(__name__)
    self._distributed = distributed
    self._output_dir = output_dir
    self._dataset_name = dataset_name
    self._do_evaluation = dataset_name.lower() in _EVAL_SETS

    if tasks is not None and isinstance(tasks, CfgNode):
      kpt_oks_sigmas = (tasks.TEST.KEYPOINT_OKS_SIGMAS
                        if not kpt_oks_sigmas else kpt_oks_sigmas)
      self._logger.warn(
          "SOT Evaluator instantiated using config, this is deprecated behavior."
          " Please pass in explicit arguments instead.")
      self._tasks = None  # Infering it from predictions should be better
    else:
      self._tasks = tasks

    self._cpu_device = torch.device("cpu")

  def reset(self):
    self._predictions = []

  def process(self, inputs, outputs):

    prediction = {"sequence": inputs, "visible": inputs.target_visible}
    if self._output_dir is not None:
      save_tracker_output(inputs.name, self._output_dir, outputs)
      # save_pth = self._output_dir / f'{inputs.name}.txt'
      # outputs['target_bbox'] = np.loadtxt(self._output_dir/f'{inputs.name}.txt')

    if "target_bbox" in outputs:
      target_bbox = torch.tensor(outputs["target_bbox"], dtype=torch.float32)
      prediction["target_bbox"] = target_bbox
    if "proposals" in outputs:
      prediction["proposals"] = outputs["proposals"].to(self._cpu_device)
    if self._do_evaluation:
      gt_boxes = inputs.ground_truth_rect
      if isinstance(gt_boxes, (dict, OrderedDict)):
        ### TODO
        gt_boxes = list(gt_boxes.values())
      prediction['gt_boxes'] = torch.tensor(gt_boxes, dtype=torch.float32)
    if len(prediction) > 1:
      self._predictions.append(prediction)

  def evaluate(self):
    if not self._do_evaluation:
      return {}
    if self._distributed:
      comm.synchronize()
      predictions = comm.gather(self._predictions, dst=0)
      predictions = list(itertools.chain(*predictions))

      if not comm.is_main_process():
        return {}
    else:
      predictions = self._predictions

    if len(predictions) == 0:
      self._logger.warning("[SOT evaluator] Did not receive valid predictions.")
      return {}

    if self._output_dir:
      file_path = Path(self._output_dir) / "target_bboxes.pth"
      with file_path.open("wb") as f:
        torch.save(predictions, f)

    self._results = OrderedDict()
    if "target_bbox" in predictions[0]:
      self._eval_tracking_boxes(predictions)
    # Copy so the caller can do whatever with results
    return copy.deepcopy(self._results)

  def _tasks_from_predictions(self, predictions):
    """
        Get COCO API "tasks" (i.e. iou_type) from COCO-format predictions.
        """
    tasks = {"bbox"}
    for pred in predictions:
      if "segmentation" in pred:
        tasks.add("segm")
      if "keypoints" in pred:
        tasks.add("keypoints")
    return sorted(tasks)

  def _eval_tracking_boxes(self, predictions):
    tasks = self._tasks or self._tasks_from_predictions(predictions)
    threshold_set_overlap = torch.arange(0.0,
                                         1.0 + 0.05,
                                         0.05,
                                         dtype=torch.float32)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float32)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float32) / 100.0

    avg_overlap_all = torch.zeros((len(predictions)), dtype=torch.float32)
    ave_success_rate_plot_overlap = torch.zeros(
        (len(predictions), threshold_set_overlap.numel()), dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros(
        (len(predictions), threshold_set_center.numel()), dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros(
        (len(predictions), threshold_set_center.numel()), dtype=torch.float32)

    # valid_sequence = torch.ones(len(predictions), dtype=torch.uint8)

    pred_boxes = [p['target_bbox'] for p in predictions]
    gt_boxes = [p['gt_boxes'] for p in predictions]
    visibles = [p.get('visible', None) for p in predictions]
    # self._calculate_metrics(pred_boxes, gt_boxes)
    for seq_id, (pred_bb, anno_bb, target_visible) in enumerate(
        zip(pred_boxes, gt_boxes, visibles)):
      # Calculate measures
      err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(
          pred_bb, anno_bb, self._dataset_name, target_visible)
      avg_overlap_all[seq_id] = err_overlap[valid_frame].mean()

      seq_length = anno_bb.shape[0]

      if seq_length <= 0:
        raise Exception('Seq length zero')

      ave_success_rate_plot_overlap[
          seq_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(
              1, -1)).sum(0).float() / seq_length * 100
      ave_success_rate_plot_center[
          seq_id, :] = (err_center.view(-1, 1) <= threshold_set_center.view(
              1, -1)).sum(0).float() / seq_length * 100
      ave_success_rate_plot_center_norm[seq_id, :] = (
          err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(
              1, -1)).sum(0).float() / seq_length * 100
    auc_curve = ave_success_rate_plot_overlap.mean(0)
    sot_results = {
        'AUC': auc_curve.mean(-1).item(),
        'OP50': auc_curve[threshold_set_overlap == 0.50].item(),
        'OP75': auc_curve[threshold_set_overlap == 0.75].item(),
        'Precision': ave_success_rate_plot_center.mean(0)[20].item(),
        'NormPrecision': ave_success_rate_plot_center_norm.mean(0)[20].item()
    }
    self._results['sot'] = sot_results
