import os
import copy
import json
import torch
import logging
import itertools
import contextlib
from collections import OrderedDict, defaultdict
from pathlib import Path
from .evaluator import DatasetEvaluator
from trackron.utils import comm
from trackron.config import CfgNode
import motmetrics as mm
from tao.toolkit.tao import TaoEval


def compare_dataframes(gts, ts):
  accs = []
  names = []
  for k, tsacc in ts.items():
    if k in gts:
      logging.info('Comparing {}...'.format(k))
      accs.append(
          mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
      names.append(k)
    else:
      logging.warning('No ground truth for {}, skipping.'.format(k))

  return accs, names


class TAOEvaluator(DatasetEvaluator):

  def __init__(self,
               dataset_name,
               distributed=True,
               output_dir=None,
               tasks=None):
    self._logger = logging.getLogger(__name__)
    self._distributed = distributed
    self._output_dir = output_dir
    self._dataset_name = dataset_name
    self._data_root = None

    self._tasks = tasks

    self._cpu_device = torch.device("cpu")

  def reset(self):
    self._predictions = []
    self._gt_file = None

  def process(self, inputs, outputs):
    """
    Args:
        inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
            It is a list of dict. Each dict corresponds to an image and
            contains keys like "height", "width", "file_name", "image_id".
        outputs: the outputs of a COCO model. It is a list of dicts with key
            "instances" that contains :class:`Instances`.
    """
    if self._gt_file is None:
      self._gt_file = inputs.coco_path
    vid_name = inputs.name
    save_path = self._output_dir / f'{vid_name}.json'
    for idx, frame_info in enumerate(outputs['frames']):
      if len(frame_info) < 1:
        continue  ### only process key frame
      video_id = frame_info['video_id']
      image_id = frame_info['id']
      prediction = {"image_id": image_id, "video_id": video_id}
      output = outputs['track_out'][idx]
      prediction["tracking_results"] = prediction_to_tao_json(
          output, video_id, image_id)
      with open(save_path, 'w') as f:
        json.dump(prediction["tracking_results"], f)

      if len(prediction) > 1:
        self._predictions.extend(prediction["tracking_results"])

  def evaluate(self):
    """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
    """
    if self._distributed:
      comm.synchronize()
      predictions = comm.gather(self._predictions, dst=0)
      predictions = list(itertools.chain(*predictions))
      if not comm.is_main_process():
        return {}
    else:
      predictions = self._predictions

    if len(predictions) == 0:
      self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
      return {}

    if self._output_dir:
      file_path = os.path.join(self._output_dir, "results.json")
      with open(file_path, "w") as f:
        json.dump(predictions, f)
        # torch.save(predictions, f)
    # logging = self._logger
    
    tao_eval = TaoEval(self._gt_file, file_path)
    tao_eval.run()
    tao_eval.print_results()

    return {}


def prediction_to_tao_json(pred_list_dict, video_id, img_id):
  """
    Dump an "Tracking" object to a TAO-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the video id
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
  results = []
  for pred_dict in pred_list_dict:
    # num_prediction = len(pred_dict['score'])
    # box = box_convert(pred)
    # bbox = box_convert(pred_dict['bbox'], 'xyxy', 'xywh').cpu().tolist()
    x1, y1, x2, y2 = pred_dict['bbox']
    bbox = [x1, y1, x2-x1, y2-y1]
    item = {
        "video_id": video_id,
        "image_id": img_id,
        "category_id": pred_dict['class'],
        "bbox": bbox,
        "score": float(pred_dict['score']),
        "track_id": int(pred_dict['tracking_id']),
    }
    results.append(item)

  return results