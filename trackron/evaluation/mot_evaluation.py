import os
import copy
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


def compare_dataframes(gts, ts):
  accs = []
  names = []
  for k, tsacc in ts.items():
    if k in gts:
      # logging.info('Comparing {}...'.format(k))
      accs.append(
          mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
      names.append(k)
    else:
      pass
      # logging.warning('No ground truth for {}, skipping.'.format(k))

  return accs, names


class MOTEvaluator(DatasetEvaluator):

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
    self._gt_files = []
    self._pred_files = []

  def process(self, inputs, outputs):
    """
    Args:
        inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
            It is a list of dict. Each dict corresponds to an image and
            contains keys like "height", "width", "file_name", "image_id".
        outputs: the outputs of a COCO model. It is a list of dicts with key
            "instances" that contains :class:`Instances`.
    """
    vid_name = inputs.name
    save_path = self._output_dir / f'{vid_name}.txt'
    if self._data_root is None:
      self._data_root = inputs.data_root
    self._gt_files += [inputs.gt_file]
    self._pred_files += [save_path]
    tracks = defaultdict(list)
    for idx, result in enumerate(outputs['track_out']):
      frame_id = idx + 1
      for item in result:
        if not ("tracking_id" in item):
          raise NotImplementedError
        tracking_id = item["tracking_id"]
        bbox = item["bbox"]  ### box in xyxy format
        bbox = [
            bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], item['score'], item['active']
        ]
        tracks[tracking_id].append([frame_id] + bbox)

    rename_track_id = 0
    with open(save_path, 'w') as f:
      for track_id in sorted(tracks):
        rename_track_id += 1
        for t in tracks[track_id]:
          if t[6] > 0:
            f.write("{},{},{:.1f},{:.1f},{:.1f},{:.1f},{:.5f},-1,-1,-1\n".format(
                t[0], rename_track_id, t[1], t[2], t[3], t[4], t[5]))

  def evaluate(self):
    """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
    """
    if self._distributed:
      comm.synchronize()
      gtfiles = comm.gather(self._gt_files, dst=0)
      gtfiles = list(itertools.chain(*gtfiles))
      tsfiles = comm.gather(self._pred_files, dst=0)
      tsfiles = list(itertools.chain(*tsfiles))
      if not comm.is_main_process() or len(gtfiles) < 1:
        return {}

    # gtfiles = [f for f in self._data_root.glob('*/gt/gt.txt')]
    # tsfiles = [ f for f in self._output_dir.glob('*.txt')]
    if gtfiles[0] is None:
      return {}

    self._logger.info('Found {} groundtruths and {} test files.'.format(
        len(gtfiles), len(tsfiles)))
    self._logger.info('Available LAP solvers {}'.format(
        mm.lap.available_solvers))
    self._logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    self._logger.info('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3],
                       mm.io.loadtxt(f, fmt='mot16', min_confidence=1))
                      for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0],
                       mm.io.loadtxt(str(f), fmt='mot16', min_confidence=-1))
                      for f in tsfiles])
    #     ts = gt

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    self._logger.info('Running metrics')
    metrics = [
        'recall', 'precision', 'num_unique_objects', 'mostly_tracked',
        'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
        'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects'
    ]
    summary = mh.compute_many(accs,
                              names=names,
                              metrics=metrics,
                              generate_overall=True)
    div_dict = {
        'num_objects': [
            'num_false_positives', 'num_misses', 'num_switches',
            'num_fragmentations'
        ],
        'num_unique_objects': [
            'mostly_tracked', 'partially_tracked', 'mostly_lost'
        ]
    }
    for divisor in div_dict:
      for divided in div_dict[divisor]:
        summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = [
        'num_false_positives', 'num_misses', 'num_switches',
        'num_fragmentations', 'mostly_tracked', 'partially_tracked',
        'mostly_lost'
    ]
    for k in change_fmt_list:
      fmt[k] = fmt['mota']
    self._logger.info('\n' + 
        mm.io.render_summary(summary,
                             formatters=fmt,
                             namemap=mm.io.motchallenge_metric_names))
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs,
                              names=names,
                              metrics=metrics,
                              generate_overall=True)
    self._logger.info('\n' +
        mm.io.render_summary(summary,
                             formatters=mh.formatters,
                             namemap=mm.io.motchallenge_metric_names))
    results = {
        'MOTA': summary['mota']['OVERALL'] * 100,
        'IDF1': summary['idf1']['OVERALL'] * 100,
        'Recall': summary['recall']['OVERALL'] * 100,
        'Precision': summary['precision']['OVERALL'] * 100
    }
    return {'MOT': results}
