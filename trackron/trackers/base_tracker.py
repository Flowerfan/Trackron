import abc
import torch
import torch.nn as nn
from torchvision.ops import box_convert
from trackron.utils import plotting


def clip_box(box: list, img_sz, margin=0):
  H, W = img_sz
  x1, y1, w, h = box
  x2, y2 = x1 + w, y1 + h
  x1 = min(max(0, x1), W - margin)
  x2 = min(max(margin, x2), W)
  y1 = min(max(0, y1), H - margin)
  y2 = min(max(margin, y2), H)
  w = max(margin, x2 - x1)
  h = max(margin, y2 - y1)
  return [x1, y1, w, h]


class BaseTracker:
  """Base class for all trackers."""

  def __init__(self, cfg, net, tracking_mode='sot'):
    self.net = net
    self.cfg = cfg
    self.visdom = None
    self.debug_level = 0
    self.frame_num = 1
    self.device = self.net.device
    self.tracking_mode = tracking_mode
    self.setup()

  @abc.abstractmethod
  def setup(self):
    pass

  def finish(self):
    return

  def predicts_segmentation_mask(self):
    return False

  def init_sot(self, image, info, **kwargs):
    """single object tracking initialization"""
    raise NotImplementedError

  def init_vos(self, image, info, **kwargs):
    """single object tracking initialization"""
    raise NotImplementedError

  def init_mot(self, image, info, **kwargs):
    """init model when tracking in mot mode"""
    raise NotImplementedError

  def initialize(self, image, info: dict, **kwargs) -> dict:
    """Overload this function in your tracker. This should initialize the model."""
    self.frame_num = 1
    if self.tracking_mode == 'sot':
      return self.init_sot(image, info, **kwargs)
    elif self.tracking_mode == 'vos':
      return self.init_vos(image, info, **kwargs)
    elif self.tracking_mode == 'mot':
      return self.init_mot(image, info, **kwargs)
    else:
      raise NotImplementedError

  def track_sot(self, image, info: dict = None) -> dict:
    """tracking in single object mode /with inital annotations"""
    raise NotImplementedError

  def track_mot(self, image, info: dict = None) -> dict:
    """tracking in multiple object mode without initial annotations"""
    raise NotImplementedError

  def track(self, image, info: dict = None, mode: str = None) -> dict:
    self.frame_num += 1
    tracking_mode = self.tracking_mode if mode is None else mode
    if tracking_mode == 'sot':
      return self.track_sot(image, info)
    if tracking_mode == 'vos':
      return self.track_vos(image, info)
    elif tracking_mode == 'mot':
      return self.track_mot(image, info)
    else:
      raise ValueError('tracking mode is not supported')

  def switch_tracking_mode(self, image, info):
    if self.tracking_mode == "mot":
      self.tracking_mode = 'sot'
      new_info = {"init_bbox": info['init_annotations'][0]['bbox']}
      frame_num = self.frame_num
      self.color = getattr(self, 'color', plotting._get_rand_color())
      self.init_sot(image, new_info)
      self.frame_num = frame_num
      if getattr(self, 'color', None) is None:
        new_info = {"init_bbox": info['init_annotations'][9]['bbox']}
        self.color = plotting._get_rand_color()
        self.init_sot(image, new_info)
        self.frame_num = frame_num
      # self.color = getattr(self, 'color', plotting._get_rand_color())
      # self.init_sot(image, new_info)
    else:
      self.tracking_mode = 'mot'

  # def visdom_draw_tracking(self, image, box, segmentation=None):
  #   if isinstance(box, OrderedDict):
  #     box = [v for k, v in box.items()]
  #   else:
  #     box = (box,)
  #   if segmentation is None:
  #     self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
  #   else:
  #     self.visdom.register((image, *box, segmentation), 'Tracking', 1,
  #                          'Tracking')

  def visdom_draw_tracking(self,
                           image,
                           out,
                           segmentation=None,
                           vwriter=None,
                           tracking_mode=None):
    tracking_mode = self.tracking_mode if tracking_mode is None else tracking_mode
    if tracking_mode == "sot":
      box = out['target_bbox']
      if self.frame_num == 1:
        self.color = plotting._get_rand_color()
        self.color = [157, 60, 153]
      box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
      image = plotting.draw_box(image.copy(),
                                box,
                                color=self.color,
                                thickness=5)
      # cv2.putText(image, "SOT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 4,
      #             (255, 0, 0), 5)
      # image = cv2.resize(image, (320, 240))
      self.visdom.register(image, 'image', 1, 'tracking')
    if tracking_mode == "mot":
      if self.frame_num == 1 or self.id_color is None:
        self.id_color = {}
      for instance in out[0]:
        if instance['active']:
          box = instance['bbox']
          track_id = instance['tracking_id']
          # box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
          text = f"ID{track_id}"
          if track_id not in self.id_color:
            self.id_color[track_id] = plotting._get_rand_color()
          image = plotting.draw_box(image.copy(),
                                    box,
                                    color=self.id_color[track_id],
                                    text=text,
                                    thickness=5)
      # cv2.putText(image, "MOT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 4,
      #             (255, 0, 0), 5)
      # image = cv2.resize(image, (480, 320))
      self.visdom.register(image, 'image', 1, 'tracking')
    if vwriter is not None:
      vwriter.writeFrame(image)
    return image

_NAME_MAP = {'pred_logits': 'scores',
             'pred_scores': 'scores',
             'pred_labels': 'labels',
             'pred_embs': 'embs',
             'tracking_boxes': 'track_boxes',
             'tracking_scores': 'track_scores',
             'tracking_logits': 'track_scores'}
class PostProcess(nn.Module):
  """ This module converts the model's output into the format expected by the coco api"""


  def __init__(self, image_sz=(608, 1088), box_fmt='cxcywh', box_absolute=False, uni_scale=False) -> None:
    super().__init__()
    self.h = float(image_sz[0])
    self.w = float(image_sz[1])
    self.box_fmt = box_fmt
    self.box_absolute = box_absolute
    self.uni_scale = uni_scale

  @torch.no_grad()
  def forward(self, outputs, origin_sizes, category=None, box_fmt='cxcywh'):
    """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            origin_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
    assert origin_sizes.shape[1] == 2
    if 'pred_classes' in outputs:
      labels = outputs['pred_classes']
      scores = outputs['class_scores']
      if 'object_scores' in outputs:
        scores = scores * outputs['object_scores']
    else:
      if 'pred_logits' in outputs:
        box_prob = outputs['pred_logits'].sigmoid()
      elif 'pred_scores' in outputs:
        box_prob = outputs['pred_scores']
      if category is None:
        scores, labels = box_prob.max(-1)
      else:
        scores, labels = box_prob[..., category:category + 1].max(-1)
    # if 'object_scores' in outputs:
    #   scores = outputs['object_scores']

    labels = labels + 1
    boxes = box_convert(outputs['pred_boxes'], self.box_fmt, 'xyxy')

    det_embs = outputs.get('pred_embs', [None] * len(scores))


    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = origin_sizes.unbind(1)
    if self.box_absolute:
      scale_fct = torch.stack([img_w / self.w, img_h / self.h, img_w / self.w, img_h / self.h], dim=1).unsqueeze(1)
    else:
      scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).unsqueeze(1)
    if self.uni_scale:
      scale_fct = scale_fct.max()
    boxes = boxes * scale_fct
    # boxes = boxes * scale_fct[:, None, :]

    #### tracking methods
    if 'tracking_boxes' in outputs:
      track_boxes = box_convert(outputs['tracking_boxes'], self.box_fmt, 'xyxy')
      track_scores = self.get_track_scores(outputs, category=category)
    else:
      track_boxes = boxes
      track_scores = scores
    track_boxes = track_boxes * scale_fct


    results = []
    for idx, score in enumerate(scores):
      entry = {'scores': score,
               'labels': labels[idx],
               'boxes': boxes[idx],
               'embs': det_embs[idx],
               'track_scores': track_scores[idx],
               'track_boxes': track_boxes[idx]}
      results.append(entry)
    return results
    # results = [{
    #     'scores': s,
    #     'labels': l,
    #     'boxes': b,
    #     'embs': emb,
    #     'track_scores': ts,
    #     'track_boxes': tb
    # } for s, l, b, emb, ts, tb in zip(scores, labels, boxes, det_embs,
    #                                   track_scores, track_boxes)]
    # return results

  def get_track_scores(self, outputs, category=None):
    assert 'tracking_logits' in outputs or 'tracking_scores' in outputs
    if outputs.get('tracking_scores', None) is not None:
      track_scores = outputs['tracking_scores'].sigmoid()
    else:
      track_prob = outputs['tracking_logits'].sigmoid()
      if category is None:
        track_scores, _ = track_prob.max(-1)
      else:
        track_scores, _ = track_prob[..., category:category + 1].max(-1)
    return track_scores


class PublicPostProcess(PostProcess):
  """process public detections"""
  ### only support person class currently

  @torch.no_grad()
  def forward(self, outputs, origin_sizes, category=None):
    assert origin_sizes.shape[1] == 2
    boxes = box_convert(outputs['pred_boxes'], 'cxcywh', 'xyxy')
    scores = outputs['pred_logits']
    labels = outputs.get('labels', torch.ones_like(scores))
    det_embs = outputs.get('pred_embs', [None] * len(boxes))
    track_boxes = box_convert(outputs['tracking_boxes'], 'cxcywh', 'xyxy')
    track_scores = self.get_track_scores(outputs, category=category)

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = origin_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    track_boxes = track_boxes * scale_fct[:, None, :]

    results = [{
        'scores': s,
        'labels': l,
        'boxes': b,
        'embs': emb,
        'track_scores': ts,
        'track_boxes': tb
    } for s, l, b, emb, ts, tb in zip(scores, labels, boxes, det_embs,
                                      track_scores, track_boxes)]
    return results