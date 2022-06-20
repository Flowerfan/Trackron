import os
import json
from PIL import Image
from pathlib import Path


def save_trackingnet_resolution():
  # dire_root = '/data/mafan/TrackingNet'
  dire_root = '/datasets01/trackingnet'
  save_file = Path.cwd() / 'trackingnet_resolution.json'
  # chunks = [0, 1, 2, 3]
  chunks = range(12)

  results = {}
  for chunk in chunks:
    dire = os.path.join(dire_root, 'TRAIN_%d' % chunk, 'frames')
    file_names = os.listdir(dire)
    for name in file_names:
      img_file = os.path.join(dire, name, '0.jpg')
      im = Image.open(img_file)
      results[name] = im.size

  json.dump(results, open(save_file, 'w'))


def save_lasot_resolution():
  dire_root = Path('/data02/mafan/LaSOTBenchmark')
  save_file = Path('./losot_resolution.json')

  results = {}

  for sub_dir in dire_root.iterdir():
    if sub_dir.is_dir():
      for seq_path in sub_dir.iterdir():
        seq_name = seq_path.name
        img_file = seq_path / 'img' / '00000001.jpg'
        im = Image.open(img_file)
        results[seq_name] = im.size
  json.dump(results, save_file.open('w', encoding="UTF-8"))

# save_lasot_resolution()
save_trackingnet_resolution()