import os
import io
import json
import datetime
import contextlib
import logging
import numpy as np
from pathlib import Path
from itertools import chain
from collections import defaultdict
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from .base_dataset import BaseDataset
# from trackron.evaluation.utils.load_text import load_text
from trackron.data import DatasetCatalog, MetadataCatalog
from trackron.structures import BoxMode, TAO, Sequence, SequenceList
"""
This file contains functions to parse TAO-format annotations into dicts in "Tracker format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_tao_json",  "register_tao"]


def nested_dict():
  return defaultdict(nested_dict)



class TAODataset(BaseDataset):
  """ TAO dataset.

    Publication:
        TAO: Tracking Any Object Dataset
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

  def __init__(self,
               data_root: Path,
               split='train',
               mode='mot'):
    super().__init__()
    # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
    self.base_path = data_root / 'frames' 
    self.ann_path = data_root / 'annotations' / f'{split}.json'

    self.dataset = json.load(open(self.ann_path))
    self.sequence_list = self._get_sequence_list()
    self.init_anns()
    self.split = split
    self.mode = mode

  def init_anns(self):
    print("Creating TAO index.")

    self.vids = {x['id']: x for x in self.dataset['videos']}
    self.tracks = {x['id']: x for x in self.dataset['tracks']}
    self.cats = {x['id']: x for x in self.dataset['categories']}

    self.imgs = {}
    self.image_name_id_map = {}
    self.vid_img_map = defaultdict(list)
    for image in self.dataset['images']:
      self.imgs[image['id']] = image
      img_path = str(self.base_path / image['file_name'])
      self.image_name_id_map[img_path] = image['id']
      self.vid_img_map[image['video_id']].append(image)

    self.vid_track_map = defaultdict(list)
    for track in self.tracks.values():
      self.vid_track_map[track['video_id']].append(track)

    self.anns = {}
    self.img_ann_map = defaultdict(list)
    self.cat_img_map = defaultdict(list)
    self.track_ann_map = defaultdict(list)
    negative_anns = []
    self.has_gt = len(self.dataset.get("annotations", [])) >= 1 
    for ann in self.dataset.get("annotations", []):
      # The category id is redundant given the track id, but we still
      # require it for compatibility with TAO tools.
      ann['bbox'] = [float(x) for x in ann['bbox']]
      if (ann['bbox'][0] < 0 or ann['bbox'][1] < 0 or ann['bbox'][2] <= 0 or
          ann['bbox'][3] <= 0):
        negative_anns.append(ann['id'])
      assert 'category_id' in ann, (f'Category id missing in annotation: {ann}')
      assert (ann['category_id'] == self.tracks[ann['track_id']]['category_id'])
      self.track_ann_map[ann['track_id']].append(ann)
      self.img_ann_map[ann["image_id"]].append(ann)
      self.cat_img_map[ann["category_id"]].append(ann["image_id"])
      self.anns[ann["id"]] = ann

    print("Index created.")

  def get_sequence_list(self):
    return SequenceList(
        [self._construct_sequence(s) for s in self.sequence_list])

  def _construct_sequence(self, vid):
    vname = self.vids[vid]['name']
    frames_path = self.base_path / vname 
    frames_list = list(sorted([str(frame) for frame in frames_path.glob('*.jpg') ]))
    min_index = min([image['frame_index'] for image in self.vid_img_map[vid]])
    init_data = nested_dict()
    object_ids = []
    bboxes = {}
    #### add frame info
    for frame_idx, frame_name in enumerate(frames_list):
      frame_eval_id = self.image_name_id_map.get(frame_name, -1)
      if frame_eval_id == -1:
        init_data[frame_idx]['image'] = {}
        init_data[frame_idx]['annotations'] = []
        init_data[frame_idx]['bbox'] = {}
        continue
      image_info = self.imgs[frame_eval_id]
      image_id = image_info['id']
      init_data[frame_idx]['image'] = image_info
      if self.has_gt:
        init_data[frame_idx]['annotations'] = self.img_ann_map[image_id]
        for ann in self.img_ann_map[image_id]:
          track_id = ann['track_id']
          object_ids += [track_id]
          init_data[frame_idx]['bbox'][track_id] = ann['bbox']
    return Sequence(vname.split('/')[-1],
                    frames_list,
                    'tao',
                    init_data,
                    init_data=init_data,
                    object_ids=object_ids,
                    multiobj_mode=True,
                    coco_path=self.ann_path,
                    tracking_mode=self.mode)

  def get_init_objects(self, track_id, init_type='first'):
    if init_type == 'first':
      return self.get_kth_annotation(track_id, k=0)
    elif init_type == 'biggest':
      return max(self.track_ann_map[track_id], key=lambda x: x['area'])
    else:
      raise NotImplementedError(f'Unsupported init type, {init_type}')

  def get_kth_annotation(self, track_id, k):
    """Return kth annotation for a track."""
    return sorted(self.track_ann_map[track_id],
                  key=lambda x: self.imgs[x['image_id']]['frame_index'])[k]

  def __len__(self):
    return len(self.sequence_list)

  def _get_sequence_list(self):
    sequence_list = [d['id'] for d in self.dataset['videos']]
    return sequence_list



def load_tao_json(json_file,
                  data_root,
                  dataset_name=None,
                  extra_annotation_keys=None):
  """
    Load a json file with TAO's video annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in TAO instances annotation format.
        data_root (str or path-like): the directory where the video images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., tao_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Tracking standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Tracker standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """

  timer = Timer()
  #   json_file = PathManager.get_local_path(json_file)
  json_file = Path(json_file)
  data_root = Path(data_root)
  with contextlib.redirect_stdout(io.StringIO()):
    tao_api = TAO(json_file)
  if timer.seconds() > 1:
    logger.info("Loading {} takes {:.2f} seconds.".format(
        json_file, timer.seconds()))

  id_map = None
  if dataset_name is not None:
    meta = MetadataCatalog.get(dataset_name)
    cat_ids = sorted(tao_api.get_cat_ids())
    cats = tao_api.load_cats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes

    # In TAO, certain category ids are artificially removed,
    # and by convention they are always ignored.
    # We deal with TAO's id issue and translate
    # the category ids to contiguous ids in [0, 1230).

    # It works by looking at the "categories" field in the json, therefore
    # if users' own json also have incontiguous ids, we'll
    # apply this mapping as well but print a warning.
    if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
      if "tao" not in dataset_name:
        logger.warning(
            """ Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you. """
        )
    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.thing_dataset_id_to_contiguous_id = id_map

  # sort indices for reproducible results
  # img_ids = sorted(tao_api.imgs.keys())
  video_ids = sorted(tao_api.vids.keys())
  # video: {
  #     "id": int,
  #     "name": str,
  #     "width" : int,
  #     "height" : int,
  #     "neg_category_ids": [int],
  #     "not_exhaustive_category_ids": [int],
  #     "metadata": dict,  # Metadata about the video
  # }
  videos = tao_api.load_vids(video_ids)
  # image: {
  #     "id" : int,
  #     "video_id": int,
  #     "file_name" : str,
  #     "license" : int,
  #     # Redundant fields for COCO-compatibility
  #     "width": int,
  #     "height": int,
  #     "frame_index": int
  # }
  video_imgs = [tao_api.vid_img_map[vid['id']] for vid in videos]
  # anns is a list[list[dict]], where each dict is an annotation
  # record for an object. The inner list enumerates the objects in an image
  # and the outer list enumerates over images. Example of anns[0]:
  # [{'segmentation': [[192.81,
  #     247.09,
  #     ...
  #     219.03,
  #     249.06]],
  #   'area': 1035.749,
  #   'iscrowd': 0,
  #   'image_id': 1268,
  #   'bbox': [192.81, 224.8, 74.73, 33.43],
  #   'category_id': 16,
  #   'id': 42986},
  #  ...]
  anns = [[tao_api.img_ann_map[img['id']] for img in imgs] for imgs in video_imgs]
  total_num_valid_anns = sum([sum([len(x) for x in xs]) for xs in anns])
  total_num_anns = len(tao_api.anns)
  if total_num_valid_anns < total_num_anns:
    logger.warning(
        f"{json_file} contains {total_num_anns} annotations, but only "
        f"{total_num_valid_anns} of them match to images in the file.")

  if not json_file.match("*minival*"):
    # The popular valminusminival & minival annotations for COCO2014 contain this bug.
    # However the ratio of buggy annotations there is tiny and does not affect accuracy.
    # Therefore we explicitly white-list them.
    ann_ids = [ann["id"] for img_ann in chain.from_iterable(anns) for ann in img_ann]
    assert len(set(ann_ids)) == len(
        ann_ids), "Annotation ids in '{}' are not unique!".format(json_file)

  num_imgs = sum([len(imgs) for imgs in video_imgs])
  logger.info("Loaded {} videos, containing {} frames, in TAO format from {}".format(
      len(videos), num_imgs, json_file))

  dataset_dicts = []

  ann_keys = ["bbox", "category_id", "track_id"
             ] + (extra_annotation_keys or [])

  num_instances_without_valid_segmentation = 0

  for (video_dict, img_dict_list, anno_dict_list) in zip(videos, video_imgs, anns):
    record = {}
    record['video_name'] = video_dict['name']
    # record["file_name"] = os.path.join(image_root, img_dict["file_name"])
    record["height"] = video_dict["height"]
    record["width"] = video_dict["width"]
    video_id = record["image_id"] = video_dict["id"]
    video_dir = data_root / video_dict['name']
    record['frame_files'] = sorted([
        x for x in video_dir.iterdir() if x.is_file() and x.suffix in ['.jpg']
    ])

    frames = {}
    tracks = defaultdict(list)
    for img_dict, anno_dict_list in zip(img_dict_list, anno_dict_list):
      image_id = img_dict['id']
      frame_index = img_dict['frame_index']
      assert img_dict['video_id'] == video_id, "video id should be the same"
      assert frame_index not in frames, "frame index should be unique for one video"
      file_name = data_root / img_dict['file_name']
      if file_name == record['frame_files'][0]:
        record['first_frame'] = img_dict

      frames[frame_index] = {"file_name": file_name}
      objs = []
      for anno in anno_dict_list:
        # Check that the image_id in this annotation is the same as
        # the image_id we're looking at.
        # This fails only when the data parsing logic or the annotation file is buggy.

        # The original TAO valminusminival2014 & minival2014 annotation files
        # actually contains bugs that, together with certain ways of using TAO API,
        # can trigger this assertion.
        assert anno["image_id"] == image_id

        assert anno.get("ignore",
                        0) == 0, '"ignore" in TAO json file is not supported.'

        obj = {key: anno[key] for key in ann_keys if key in anno}
        tracks[anno['track_id']] += [frame_index]

        segm = anno.get("segmentation", None)
        if segm:  # either list[list[float]] or dict(RLE)
          if isinstance(segm, dict):
            if isinstance(segm["counts"], list):
              # convert to compressed RLE
              segm = mask_util.frPyObjects(segm, *segm["size"])
          else:
            # filter out invalid polygons (< 3 points)
            segm = [
                poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6
            ]
            if len(segm) == 0:
              num_instances_without_valid_segmentation += 1
              continue  # ignore this instance
            obj["segmentation"] = segm
        objs.append(obj)
      frames[frame_index]["annotations"] = objs
    record["frames"] = frames
    record["tracks"] = tracks
    dataset_dicts.append(record)

  if num_instances_without_valid_segmentation > 0:
    logger.warning(
        "Filtered out {} instances without valid segmentation. ".format(
            num_instances_without_valid_segmentation) +
        "There might be issues in your dataset generation process.  Please "
        "check https://tracker.readthedocs.io/en/latest/tutorials/datasets.html carefully"
    )
  return dataset_dicts


def convert_to_tao_dict(dataset_name):
  """
    Convert an instance detection/segmentation or keypoint detection dataset
    in tracker's standard format into TAO json format.

    Generic dataset description can be found here:
    https://tracker.readthedocs.io/tutorials/datasets.html#register-a-dataset

    TAO data format description can be found here:
    http://taodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in tracker's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        tao_dict: serializable dict in TAO json format
    """

  dataset_dicts = DatasetCatalog.get(dataset_name)
  metadata = MetadataCatalog.get(dataset_name)

  # unmap the category mapping ids for TAO
  if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
    reverse_id_mapping = {
        v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
    }
    reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id
                                                                ]  # noqa
  else:
    reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

  categories = [{
      "id": reverse_id_mapper(id),
      "name": name
  } for id, name in enumerate(metadata.thing_classes)]

  logger.info("Converting dataset dicts into TAO format")
  tao_images = []
  tao_annotations = []

  for image_id, image_dict in enumerate(dataset_dicts):
    tao_image = {
        "id": image_dict.get("image_id", image_id),
        "width": int(image_dict["width"]),
        "height": int(image_dict["height"]),
        "file_name": str(image_dict["file_name"]),
    }
    tao_images.append(tao_image)

    anns_per_image = image_dict.get("annotations", [])
    for annotation in anns_per_image:
      # create a new dict with only TAO fields
      tao_annotation = {}

      # TAO requirement: XYWH box format for axis-align and XYWHA for rotated
      bbox = annotation["bbox"]
      if isinstance(bbox, np.ndarray):
        if bbox.ndim != 1:
          raise ValueError(
              f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
        bbox = bbox.tolist()
      if len(bbox) not in [4, 5]:
        raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
      from_bbox_mode = annotation["bbox_mode"]
      to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
      bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

      # TAO requirement: instance area
      if "segmentation" in annotation:
        # Computing areas for instances by counting the pixels
        segmentation = annotation["segmentation"]
        # TODO: check segmentation type: RLE, BinaryMask or Polygon
        if isinstance(segmentation, list):
          polygons = PolygonMasks([segmentation])
          area = polygons.area()[0].item()
        elif isinstance(segmentation, dict):  # RLE
          area = mask_util.area(segmentation).item()
        else:
          raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
      else:
        # Computing areas using bounding boxes
        if to_bbox_mode == BoxMode.XYWH_ABS:
          bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
          area = Boxes([bbox_xy]).area()[0].item()
        else:
          area = RotatedBoxes([bbox]).area()[0].item()

      if "keypoints" in annotation:
        keypoints = annotation["keypoints"]  # list[int]
        for idx, v in enumerate(keypoints):
          if idx % 3 != 2:
            # TAO's segmentation coordinates are floating points in [0, H or W],
            # but keypoint coordinates are integers in [0, H-1 or W-1]
            # For TAO format consistency we substract 0.5
            # https://github.com/facebookresearch/tracker/pull/175#issuecomment-551202163
            keypoints[idx] = v - 0.5
        if "num_keypoints" in annotation:
          num_keypoints = annotation["num_keypoints"]
        else:
          num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

      # TAO requirement:
      #   linking annotations to images
      #   "id" field must start with 1
      tao_annotation["id"] = len(tao_annotations) + 1
      tao_annotation["image_id"] = tao_image["id"]
      tao_annotation["bbox"] = [round(float(x), 3) for x in bbox]
      tao_annotation["area"] = float(area)
      tao_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
      tao_annotation["category_id"] = int(
          reverse_id_mapper(annotation["category_id"]))

      # Add optional fields
      if "keypoints" in annotation:
        tao_annotation["keypoints"] = keypoints
        tao_annotation["num_keypoints"] = num_keypoints

      if "segmentation" in annotation:
        seg = tao_annotation["segmentation"] = annotation["segmentation"]
        if isinstance(seg, dict):  # RLE
          counts = seg["counts"]
          if not isinstance(counts, str):
            # make it json-serializable
            seg["counts"] = counts.decode("ascii")

      tao_annotations.append(tao_annotation)

  logger.info(
      "Conversion finished, "
      f"#images: {len(tao_images)}, #annotations: {len(tao_annotations)}")

  info = {
      "date_created": str(datetime.datetime.now()),
      "description": "Automatically generated TAO json file for Tracker.",
  }
  tao_dict = {
      "info": info,
      "images": tao_images,
      "categories": categories,
      "licenses": None
  }
  if len(tao_annotations) > 0:
    tao_dict["annotations"] = tao_annotations
  return tao_dict


def register_tao(name, metadata, json_file, image_root):
  """
    Register a dataset in TAO's json annotation format for
    object tracking.

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "tao_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
  assert isinstance(name, str), name
  assert isinstance(json_file, (str, os.PathLike)), json_file
  assert isinstance(image_root, (str, os.PathLike)), image_root
  # 1. register a function which returns dicts
  DatasetCatalog.register(name,
                          lambda: load_tao_json(json_file, image_root, name))

  # 2. Optionally, add metadata about this dataset,
  # since they might be useful in evaluation, visualization or logging
  MetadataCatalog.get(name).set(json_file=json_file,
                                image_root=image_root,
                                evaluator_type="tao",
                                **metadata)


if __name__ == "__main__":
  """
    Test the TAO json dataset loader.

    Usage:
        python -m tracker.data.datasets.tao \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "tao_2014_minival_100", or other
        pre-registered ones
    """
  import trackron.data.datasets.testsets  # noqa # add pre-defined metadata
  import sys
  from PIL import Image

  assert sys.argv[3] in DatasetCatalog.list()
  meta = MetadataCatalog.get(sys.argv[3])
  sys.path.append("/home/mafan/projects/tracking")

  dicts = load_tao_json(sys.argv[1], sys.argv[2], sys.argv[3])
  logger.info("Done loading {} samples.".format(len(dicts)))

  dirname = "tao-data-vis"
  os.makedirs(dirname, exist_ok=True)
  for video in dicts:
    for frame_index, img_dict in video['frames'].items():
      print(img_dict)

      img = np.array(Image.open(img_dict["file_name"]))
      import pdb;pdb.set_trace()
      # visualizer = Visualizer(img, metadata=meta)
      # vis = visualizer.draw_dataset_dict(d)
      # fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
      # vis.save(fpath)
