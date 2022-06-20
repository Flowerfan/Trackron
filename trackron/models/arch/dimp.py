import torch
import torch.nn as nn
from typing import Dict, Tuple
from collections import OrderedDict
from trackron.config import configurable

from .build import META_ARCH_REGISTRY, BaseModel
from trackron.models.backbone import build_backbone
from trackron.models.box_heads import build_box_head
from trackron.models.cls_heads import build_cls_head
from trackron.models.objectives import build_objective, BaseObjective


@META_ARCH_REGISTRY.register()
class DiMPNet(BaseModel):

  @configurable
  def __init__(
      self,
      *,
      classification_layers: list,
      output_layers: list,
      backbone: nn.Module,
      cls_head: nn.Module,
      box_head: nn.Module,
      objective: BaseObjective,
      pixel_mean: Tuple[float],
      pixel_std: Tuple[float],
  ):
    """[summary]

    Args:
        classification_layers (list): [description]
        output_layers (list): [description]
        backbone (nn.Module): [description]
        cls_head (nn.Module): [description]
        box_head (nn.Module): [description]
        pixel_mean (Tuple[float]): [description]
        pixel_std (Tuple[float]): [description]
    """
    super().__init__()
    self.output_layers = output_layers
    self.classification_layers = classification_layers
    self.backbone = backbone
    self.cls_head = cls_head
    self.box_head = box_head
    self.objective = objective
    self.register_buffer("pixel_mean",
                         torch.tensor(pixel_mean).view(1, -1, 1, 1), False)
    self.register_buffer("pixel_std",
                         torch.tensor(pixel_std).view(1, -1, 1, 1), False)
    assert (self.pixel_mean.shape == self.pixel_std.shape
           ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

  @classmethod
  def from_config(cls, cfg):
    backbone = build_backbone(cfg)
    return {
        "backbone": backbone,
        "cls_head": build_cls_head(cfg.MODEL.SOT.CLS_HEAD),
        "box_head": build_box_head(cfg.MODEL.SOT.BOX_HEAD),
        "objective": build_objective(cfg.SOT.OBJECTIVE),
        "output_layers": cfg.MODEL.BACKBONE.OUTPUT_LAYERS,
        "classification_layers": cfg.MODEL.BACKBONE.CLS_LAYERS,
        # "input_format": cfg.INPUT.FORMAT,
        # "vis_period": cfg.VIS_PERIOD,
        "pixel_mean": cfg.MODEL.PIXEL_MEAN,
        "pixel_std": cfg.MODEL.PIXEL_STD,
    }

  def forward_sot(self, data: Dict[str, torch.Tensor]):
    # if not self.training:
    # def forward(self, template_imgs, search_imgs, template_bb, search_proposals, *args, **kwargs):
    """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            template_imgs:  Train image samples (images, sequences, 3, H, W).
            search_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            search_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            search_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the search_proposals."""

    # assert template_imgs.dim() == 5 and search_imgs.dim() == 5, 'Expect 5 dimensional inputs'
    data = self.preprocess(data)
    template_imgs, search_imgs = data['template_images'], data['search_images']
    train_bb, search_bb = data['template_boxes'], data['search_boxes']
    search_proposals = data['search_proposals']
    # Extract backbone features
    template_feat = self.extract_backbone_features(
        template_imgs.reshape(-1, *template_imgs.shape[-3:]))
    search_feat = self.extract_backbone_features(
        search_imgs.reshape(-1, *search_imgs.shape[-3:]))

    # Classification features
    template_feat_clf = self.get_backbone_clf_feat(template_feat)
    search_feat_clf = self.get_backbone_clf_feat(search_feat)

    # Run classifier module
    target_scores = self.cls_head(template_feat_clf, search_feat_clf, train_bb,
                                  search_bb)

    # Get bb_regressor features
    template_feat_iou = self.get_backbone_bbreg_feat(template_feat)
    search_feat_iou = self.get_backbone_bbreg_feat(search_feat)

    # Run the IoUNet module
    box_pred = self.box_head(template_feat_iou, search_feat_iou, train_bb,
                             search_proposals)

    out = {'target_scores': target_scores, 'box_pred': box_pred}
    return self.objective(out, data)

  def get_backbone_clf_feat(self, backbone_feat):
    feat = OrderedDict(
        {l: backbone_feat[l] for l in self.classification_layers})
    if len(self.classification_layers) == 1:
      return feat[self.classification_layers[0]]
    return feat

  @property
  def device(self):
    return self.pixel_mean.device

  def preprocess(self, data):
    data = data.to(self.device)
    data['template_images'] = self.preprocess_image(data['template_images'])
    data['search_images'] = self.preprocess_image(data['search_images'])
    # data['template_images'] = (data['template_images'] -
    #                            self.pixel_mean) / self.pixel_std
    # data['search_images'] = (data['search_images'] -
    #                          self.pixel_mean) / self.pixel_std
    # data = [(x - self.pixel_mean) / self.pixel_std for x in data['template_images']]
    # data['search_images'] =  [
    #     (x - self.pixel_mean) / self.pixel_std for x in data['search_images']

    # data['template_images'] = torch.data['template_images']
    return data

  def preprocess_image(self, im:torch.Tensor):
    if len(im.shape) == 5:
      return (im - self.pixel_mean[None]) / self.pixel_std[None]
    else:
      return (im - self.pixel_mean) / self.pixel_std

  def get_backbone_bbreg_feat(self, backbone_feat):
    return [backbone_feat[l] for l in self.output_layers]

  def extract_classification_feat(self, backbone_feat):
    return self.cls_head.extract_classification_feat(
        self.get_backbone_clf_feat(backbone_feat))

  def extract_backbone_features(self, im, layers=None):
    if layers is None:
      layers = self.output_layers
    return self.backbone(im, layers)

  def extract_features(self, im, layers=None):
    if layers is None:
      layers = self.bb_regressor_layer + ['classification']
    if 'classification' not in layers:
      return self.feature_extractor(im, layers)
    backbone_layers = sorted(
        list(
            set([
                l for l in layers + self.classification_layers
                if l != 'classification'
            ])))
    all_feat = self.feature_extractor(im, backbone_layers)
    all_feat['classification'] = self.extract_classification_feat(all_feat)
    return OrderedDict({l: all_feat[l] for l in layers})
