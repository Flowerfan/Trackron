import os
from random import choice
import shutil
import argparse
import numpy as np
from pathlib import Path


parser = argparse.ArgumentParser(
    description=""" processing tracking results for GOT10k and TrackingNet online submission file """,
      formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--results-dir',
                    type=str,
                    help='tracking results directoru')
parser.add_argument('--outputs-dir',
                    type=str,
                    help='directory for saving results')
parser.add_argument('--dataset',
                    type=str,
                    default='got10k',
                    choices=['got10k', 'trackingnet'],
                    help='processing dataset default is got10k')
args = parser.parse_args()

def pack_got10k_results(results_dir: Path, output_dir: Path):
  """ Packs got10k results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder 
    """
  if not output_dir.exists():
    output_dir.mkdir(parents=True)

  for i in range(1, 181):
    seq_name = 'GOT-10k_Test_{:06d}'.format(i)
    seq_output_path = output_dir / seq_name
    if not seq_output_path.exists():
      seq_output_path.mkdir()

    res = np.loadtxt(results_dir / f'{seq_name}.txt', dtype=np.float64)
    times = np.loadtxt(results_dir / f'{seq_name}_time.txt', dtype=np.float64)

    np.savetxt(seq_output_path / f'{seq_name}_001.txt',
               res,
               delimiter=',',
               fmt='%f')
    np.savetxt(seq_output_path / f'{seq_name}_time.txt',
               times,
               fmt='%f')

  # Generate ZIP file
  shutil.make_archive('got10k', 'zip', output_dir)

  # Remove raw text files
  shutil.rmtree(output_dir)


def pack_trackingnet_results(results_dir: Path, output_dir: Path):
  """ Packs trackingnet results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder env_settings().tn_packed_results_path
    """
  if not output_dir.exists():
    output_dir.mkdir(parents=True)

  for result_path in  results_dir.glob('*[0-9].txt'):
    results = np.loadtxt(result_path, dtype=np.float64)
    np.savetxt(output_dir / result_path.name,
               results,
               delimiter=',',
               fmt='%.2f')

  # Generate ZIP file
  shutil.make_archive('trackingnet', 'zip', output_dir)

  # Remove raw text files
  shutil.rmtree(output_dir)


results_dir = Path(args.results_dir)
outputs_dir = Path(args.outputs_dir)
if args.dataset == 'got10k':
  pack_got10k_results(results_dir, outputs_dir)
elif args.dataset == 'trackingnet':
  pack_trackingnet_results(results_dir, outputs_dir)
