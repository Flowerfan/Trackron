import os
import numpy as np
import json
import cv2
import argparse
import pycocotools.mask as rletools
from collections import defaultdict
from pathlib import Path
import configparser


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='', help='data root contain MOT datasets')
args = parser.parse_args()
# Use the same script for MOT16
DATA_PATH = Path(args.data_root)
# DATA_PATH = Path('/data/mafan/MOT/MOTS')
OUT_PATH = DATA_PATH / 'annotations'
SPLITS = ['train_half', 'val_half', 'train', 'test']
# SPLITS = ['val_half']
HALF_VIDEO = True
PUBLIC_DETECTION = 'FRCNN'
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True
MOTS = 'MOTS' in str(DATA_PATH)


if not OUT_PATH.exists():
  OUT_PATH.mkdir()


class SegmentedObject:
  """
    Helper class for segmentation objects.
    """

  def __init__(self, mask: dict, class_id: int, track_id: int) -> None:
    self.mask = mask
    self.class_id = class_id
    self.track_id = track_id


def load_mots_gt(path: Path) -> dict:
  """Load MOTS ground truth from path."""
  objects_per_frame = {}
  track_ids_per_frame = {
  }  # Check that no frame contains two objects with same id
  combined_mask_per_frame = {}  # Check that no frame contains overlapping masks

  with path.open("r") as gt_file:
    for line in gt_file:
      line = line.strip()
      fields = line.split(" ")

      frame = int(fields[0])
      if frame not in objects_per_frame:
        objects_per_frame[frame] = []
      if frame not in track_ids_per_frame:
        track_ids_per_frame[frame] = set()
      if int(fields[1]) in track_ids_per_frame[frame]:
        assert False, f"Multiple objects with track id {fields[1]} in frame {fields[0]}"
      else:
        track_ids_per_frame[frame].add(int(fields[1]))

      class_id = int(fields[2])
      if not (class_id == 1 or class_id == 2 or class_id == 10):
        assert False, "Unknown object class " + fields[2]

      mask = {
          'size': [int(fields[3]), int(fields[4])],
          'counts': fields[5].encode(encoding='UTF-8')
      }
      if frame not in combined_mask_per_frame:
        combined_mask_per_frame[frame] = mask
      elif rletools.area(
          rletools.merge([combined_mask_per_frame[frame], mask],
                         intersect=True)):
        assert False, "Objects with overlapping masks in frame " + fields[0]
      else:
        combined_mask_per_frame[frame] = rletools.merge(
            [combined_mask_per_frame[frame], mask], intersect=False)
      objects_per_frame[frame].append(
          SegmentedObject(mask, class_id, int(fields[1])))

  return objects_per_frame


if __name__ == '__main__':
  for split in SPLITS:
    data_path = DATA_PATH / ('test' if split == 'test' else 'train')
    out_path = OUT_PATH / '{}.json'.format(split)
    out = {
        'images': [],
        'annotations': [],
        'categories': [{
            'id': 1,
            'name': 'person'
        }],
        'videos': [],
        'tracks': []
    }
    seqs = [seq.name for seq in data_path.iterdir()]
    config = configparser.ConfigParser()
    # seqs = os.listdir(data_path)
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    track_cnt = 0
    vid_track_id = defaultdict(dict)
    tracks = {}
    for seq in sorted(seqs):
      if '.DS_Store' in str(seq):
        continue
      if (split != 'test' and not (PUBLIC_DETECTION in str(seq))):
        continue
      video_cnt += 1
      config.read(data_path / seq / 'seqinfo.ini')
      imgH, imgW = int(config['Sequence']['imHeight']), int(
          config['Sequence']['imWidth'])
      fps = int(config['Sequence']['frameRate'])
      out['videos'].append({
          'id': video_cnt,
          'name': str(seq),
          'width': imgW,
          'height': imgH,
          'fps': fps
      })
      seq_path = data_path / seq
      img_path = seq_path / 'img1'
      ann_path = seq_path / 'gt/gt.txt'
      det_path = seq_path / 'det/det.txt'
      # images = img_path.iterdir()
      num_images = len(
          [image for image in img_path.iterdir() if image.suffix == '.jpg'])
      if HALF_VIDEO and ('half' in split):
        image_range = [0, num_images // 2] if 'train' in split else \
          [num_images // 2 + 1, num_images - 1]
      else:
        image_range = [0, num_images - 1]
      for i in range(num_images):
        if (i < image_range[0] or i > image_range[1]):
          continue
        image_info = {'file_name':  '{}/img1/{:06d}.jpg'.format(seq, i + 1),
                      'id': image_cnt + i + 1,
                      'frame_index': i,
                      'frame_id': i + 1 - image_range[0],
                      'width': imgW, 'height': imgH,
                      'prev_image_id': image_cnt + i if i > 0 else -1,
                      'next_image_id':                     \
                        image_cnt + i + 2 if i < num_images - 1 else -1,
                      'video_id': video_cnt}
        out['images'].append(image_info)
      print('{}: {} images'.format(seq, num_images))
      if split != 'test':
        if MOTS:
          #### For MOTS 20
          mask_objects_per_frame = load_mots_gt(ann_path)
          seq_track_ids = {}
          for frame_id, mask_objects in mask_objects_per_frame.items():
            image_id =  image_cnt + frame_id
            if image_id is None:
              continue
            for mask_object in mask_objects:
              # class_id = 1 is car
              # class_id = 2 is pedestrian
              # class_id = 10 IGNORE
              if mask_object.class_id != 2:
                continue

              bbox = rletools.toBbox(mask_object.mask)
              bbox = [int(c) for c in bbox]
              area = bbox[2] * bbox[3]

              segmentation = {
                  'size': mask_object.mask['size'],
                  'counts': mask_object.mask['counts'].decode(encoding='UTF-8')
              }
              category_id = out['categories'][0]['id']
              tgt_track_id = mask_object.track_id

              if tgt_track_id not in seq_track_ids:
                track_cnt = track_cnt + 1
                seq_track_ids[tgt_track_id] = track_cnt
                tracks[track_cnt] = {
                    'id': track_cnt,
                    'video_id': video_cnt,
                    'category_id': category_id
                }
              track_id = seq_track_ids[tgt_track_id]


              ann = {
                  "id": ann_cnt,
                  "bbox": bbox,
                  "image_id": image_id,
                  "segmentation": segmentation,
                  "ignore": mask_object.class_id == 10,
                  "visibility": 1.0,
                  "area": area,
                  "iscrowd": 0,
                  "seq": seq,
                  "category_id": category_id,
                  "track_id": track_id
              }

              out['annotations'].append(ann)
              ann_cnt += 1

        else:
          det_path = seq_path / 'det/det.txt'
          anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
          dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
          # if CREATE_SPLITTED_ANN and ('half' in split):
          if CREATE_SPLITTED_ANN:
            anns_out = np.array([
                anns[i] for i in range(anns.shape[0]) if int(anns[i][0]) -
                1 >= image_range[0] and int(anns[i][0]) - 1 <= image_range[1]
            ], np.float32)
            anns_out[:, 0] -= image_range[0]
            gt_out = seq_path / 'gt/gt_{}.txt'.format(split)
            fout = open(gt_out, 'w')
            for o in anns_out:
              fout.write(
                  '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                      int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]),
                      int(o[5]), int(o[6]), int(o[7]), o[8]))
            fout.close()
          if CREATE_SPLITTED_DET and ('half' in split):
            dets_out = np.array([dets[i] for i in range(dets.shape[0]) if \
              int(dets[i][0]) - 1 >= image_range[0] and \
              int(dets[i][0]) - 1 <= image_range[1]], np.float32)
            dets_out[:, 0] -= image_range[0]
            det_out = seq_path / 'det/det_{}.txt'.format(split)
            dout = open(det_out, 'w')
            for o in dets_out:
              dout.write(
                  '{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                      int(o[0]), int(o[1]), float(o[2]), float(o[3]),
                      float(o[4]), float(o[5]), float(o[6])))
            dout.close()

          print(' {} ann images'.format(int(anns[:, 0].max())))
          for i in range(anns.shape[0]):
            frame_id = int(anns[i][0])
            if (frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]):
              continue
            cat_id = int(anns[i][7])
            ann_cnt += 1
            obj_id = int(anns[i][1])
            if not ('15' in str(DATA_PATH)):
              if not (float(anns[i][8]) >= 0.25):
                continue
              if not (int(anns[i][6]) == 1):
                continue
              if (int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]):  # Non-person
                continue
              if (int(anns[i][7]) in [2, 7, 8, 12]):  # Ignored person
                category_id = -1
              else:
                category_id = 1
            else:
              category_id = 1
            if video_cnt not in vid_track_id:
              track_cnt += 1
              track_id = track_cnt
              vid_track_id[video_cnt][obj_id] = track_id
            elif obj_id not in vid_track_id[video_cnt]:
              track_cnt += 1
              track_id = track_cnt
              vid_track_id[video_cnt][obj_id] = track_id
            ann = {
                'id': ann_cnt,
                'category_id': category_id,
                'image_id': image_cnt + frame_id,
                'track_id': track_id,
                'video_id': video_cnt,
                'bbox': anns[i][2:6].tolist(),
                'conf': float(anns[i][6]),
                'iscrowd': 0,
                'area': float(anns[i][4] * anns[i][5])
            }
            out['annotations'].append(ann)
            if track_id not in tracks:
              tracks[track_id] = {
                  'id': track_id,
                  'video_id': video_cnt,
                  'category_id': category_id
              }
          track_cnt = track_id
      image_cnt += num_images
    out['tracks'] = list(tracks.values())
    print('loaded {} for {} images and {} samples'.format(
        split, len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
