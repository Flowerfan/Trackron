import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from typing import Optional
from einops import rearrange
from torchvision.ops import box_convert

from trackron.config import configurable
from trackron.models.extractor import (DeformableTransformerEncoderLayer,
                                       DeformableTransformerEncoder,
                                       DeformableTransformerDecoderLayer,
                                       DeformableTransformerDecoder)
from trackron.models.layers.position_embedding import build_position_encoding
from trackron.utils.misc import inverse_sigmoid
from trackron.structures import NestedTensor
from trackron.models.box_heads import build_box_head
from trackron.models.poolers import ROIPooler

from .build import MOT_HEAD_REGISTRY


def _get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@MOT_HEAD_REGISTRY.register()
class DFDETR_PROPOSAL(nn.Module):

  @configurable
  def __init__(self,
               *,
               feature_layers: list,
               num_feature_levels: int,
               pos_emb: nn.Module,
               reference_points: nn.Module,
               encoder: nn.Module,
               decoder: nn.Module,
               pooler: nn.Module,
               input_proj: nn.Module,
               box_head: nn.Module,
               class_head: nn.Module,
               track_head: nn.Module,
               level_embed: nn.Module,
               query_emb: Optional[nn.Module] = None):
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
    self.feature_layers = feature_layers
    self.num_feature_levels = num_feature_levels
    self.pos_emb = pos_emb
    self.input_proj = input_proj
    self.query_embed = query_emb
    self.reference_points = reference_points
    self.encoder = encoder
    self.decoder = decoder
    self.pooler = pooler
    self.box_head = box_head
    self.level_embed = level_embed
    self.class_head = class_head
    self.track_head = track_head

  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    feature_layers = cfg.FEATURE_LAYERS
    num_feature_levels = cfg.NUM_FEATURE_LEVELS
    hidden_dim = cfg.FEATURE_DIM
    num_classes = cfg.NUM_CLASS
    pooler_resolution = cfg.TRACK_HEAD.POOL_SIZE
    pooler_scales = cfg.TRACK_HEAD.POOL_SCALES
    sampling_ratio = cfg.TRACK_HEAD.POOL_SAMPLE_RATIO
    pooler_type = cfg.TRACK_HEAD.POOL_TYPE
    pos_emb = build_position_encoding(cfg.POSITION_EMBEDDING, hidden_dim)
    out_channels = [in_feature_channels[layer] for layer in feature_layers]

    input_proj = [
        nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        ) for in_channels in out_channels
    ]
    if num_feature_levels > len(feature_layers):
      input_proj += [
          nn.Sequential(
              nn.Conv2d(out_channels[-1],
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1),
              nn.GroupNorm(32, hidden_dim),
          )
      ]
    input_proj = nn.ModuleList(input_proj)

    ### For MOT
    encoder_layer = DeformableTransformerEncoderLayer(
        hidden_dim, cfg.ENCODER.DIM_FEEDFORWARD, cfg.ENCODER.DROPOUT,
        cfg.ENCODER.NORM, num_feature_levels, cfg.ENCODER.HEADS,
        cfg.ENCODER.NUM_POINTS)
    encoder = DeformableTransformerEncoder(encoder_layer,
                                           cfg.ENCODER.NUM_LAYERS)
    decoder_layer = DeformableTransformerDecoderLayer(
        hidden_dim, cfg.DECODER.DIM_FEEDFORWARD, cfg.DECODER.DROPOUT,
        cfg.DECODER.NORM, num_feature_levels, cfg.DECODER.HEADS,
        cfg.DECODER.NUM_POINTS)
    decoder = DeformableTransformerDecoder(decoder_layer,
                                           cfg.DECODER.NUM_LAYERS,
                                           True)

    pooler = ROIPooler(
        output_size=pooler_resolution,
        scales=pooler_scales,
        sampling_ratio=sampling_ratio,
        pooler_type=pooler_type,
    )

    track_head = build_box_head(cfg.TRACK_HEAD)
    box_head = build_box_head(cfg.BOX_HEAD)
    class_head = nn.Linear(hidden_dim, cfg.NUM_CLASS)
    class_head.bias.data = -torch.ones(num_classes) * 4.6
    num_pred = cfg.DECODER.NUM_LAYERS
    num_queries = cfg.NUM_QUERIES
    if cfg.BOX_REFINE:
      class_head = _get_clones(class_head, num_pred)
      box_head = _get_clones(box_head, num_pred)
      decoder.bbox_embed = box_head
    else:
      class_head = nn.ModuleList([class_head for _ in range(num_pred)])
      box_head = nn.ModuleList([box_head for _ in range(num_pred)])
    query_embeds = nn.Embedding(num_queries, hidden_dim * 2)
    level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
    reference_points = nn.Linear(hidden_dim, 2)

    return {
        "query_emb": query_embeds,
        "feature_layers": feature_layers,
        "num_feature_levels": num_feature_levels,
        "pos_emb": pos_emb,
        "input_proj": input_proj,
        "reference_points": reference_points,
        "encoder": encoder,
        "decoder": decoder,
        "pooler": pooler,
        "track_head": track_head,
        "box_head": box_head,
        "class_head": class_head,
        "level_embed": level_embed,
    }

  def forward(self, temp_feat, search_feat, temp_mask, search_mask):
    """[summary]

    Args:
        temp_feat ([type]): [description]
        search_feat ([type]): [description]
        temp_mask ([type]): [description]
        search_mask ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Extract backbone features
    temp_srcs, temp_masks, temp_posembs = self.get_mot_features(
        temp_feat, temp_mask)
    search_srcs, search_masks, search_posembs = self.get_mot_features(
        search_feat, search_mask)

    ### flatten feats for attention
    spatial_shapes = torch.tensor([list(src.shape[-2:]) for src in temp_srcs],
                                  device=temp_srcs[0].device,
                                  dtype=torch.long)
    valid_ratios = torch.stack([self.get_valid_ratio(m) for m in temp_masks], 1)
    temp_srcs, temp_masks, temp_posembs = self.flatten_feats(
        temp_srcs, temp_masks, temp_posembs)
    search_srcs, search_masks, search_posembs = self.flatten_feats(
        search_srcs, search_masks, search_posembs)

    # encoder
    temp_memory = self.encoder(temp_srcs, spatial_shapes, valid_ratios, temp_posembs, temp_masks)
    search_memory = self.encoder(search_srcs, spatial_shapes, valid_ratios, search_posembs, search_masks)


    ### detection
    temp_target_emb, init_ref, inter_ref = self.decode_targets(
        temp_memory, temp_masks, spatial_shapes, valid_ratios)
    temp_logits, temp_boxes = self.decode_mot_cls_box(temp_target_emb, init_ref, inter_ref)

    #### track
    # target_feat = self.extract_target_feat(temp_memory, temp_boxes[-1], spatial_shapes)
    search_feat = self.unflatten_features(search_memory, spatial_shapes)
    proposals = self.get_proposals(temp_boxes[-1], spatial_shapes, valid_ratios)
    track_boxes, track_centers, _ = self.track_head(search_feat, temp_target_emb[-1], proposals)
    track_boxes = self.postprocess_boxes(track_boxes, spatial_shapes,
                                         valid_ratios)

    out = {'pred_logits': temp_logits, 'pred_boxes': temp_boxes,
           'track_logits': temp_logits.detach(), 'track_boxes': track_boxes}
    if len(track_centers) > 0:
      out['track_centers'] = track_centers
    return out

  def decode_targets(self, features, masks, spatial_shapes, valid_ratios):
    b, k, c = features.shape
    query_embed = self.query_embed.weight

    query_embed, tgt = torch.split(query_embed, c, dim=1)
    query_embed = query_embed.unsqueeze(0).expand(b, -1, -1)
    tgt = tgt.unsqueeze(0).expand(b, -1, -1)
    reference_points = self.reference_points(query_embed).sigmoid()
    init_reference_out = reference_points
    # decoder
    hs, inter_references = self.decoder(tgt, reference_points, features,
                                        spatial_shapes, valid_ratios,
                                        query_embed, masks)
    inter_references_out = inter_references

    return hs, init_reference_out, inter_references_out

  def get_proposals(self, boxes, spatial_shapes, valid_ratios):
    scale = spatial_shapes[0][None] * valid_ratios[:, 0] * 8
    scale = scale.flip(-1).repeat(1, 2).unsqueeze(1)
    boxes = box_convert(boxes * scale, 'cxcywh', 'xyxy')
    return boxes.detach()

  def postprocess_boxes(self, pred_boxes, spatial_shapes, valid_ratios):
    scale = spatial_shapes[0][None] * valid_ratios[:, 0] * 8
    scale = scale.flip(-1).repeat(1, 2).unsqueeze(1)
    pred_boxes = [box_convert(boxes / scale, 'xyxy', 'cxcywh') for boxes in pred_boxes]
    return pred_boxes


  def get_feature_maskposemb(self, feat, mask):
    C, H, W = feat.shape[1:]
    mask = F.interpolate(mask.float(), size=(H, W)).to(torch.bool).flatten(0, 1)
    nested_feat = NestedTensor(feat, mask)
    pos_emb = self.pos_emb(nested_feat).to(feat.dtype)
    return mask, pos_emb

  def get_mot_features(self, features, feature_masks):
    srcs = []
    masks = []
    poses = []
    for idx, layer in enumerate(features.keys()):
      feat = self.input_proj[idx](features[layer])
      mask, pos = self.get_feature_maskposemb(feat, feature_masks[None])
      srcs += [feat]
      masks += [mask]
      poses += [pos]

    if len(self.input_proj) > len(features):
      srcs += [self.input_proj[-1](features[layer])]
      mask, pos = self.get_feature_maskposemb(srcs[-1], feature_masks[None])
      masks += [mask]
      poses += [pos]
    return srcs, masks, poses

  def get_valid_ratio(self, mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio

  def flatten_feats(self, srcs, masks, pos_embeds):
    src_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
      src = src.flatten(2).transpose(1, 2)
      mask = mask.flatten(1)
      pos_embed = pos_embed.flatten(2).transpose(1, 2)
      lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
      lvl_pos_embed_flatten.append(lvl_pos_embed)
      src_flatten.append(src)
      mask_flatten.append(mask)
    src_flatten = torch.cat(src_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    return src_flatten, mask_flatten, lvl_pos_embed_flatten

  def unflatten_features(self, features, spatial_shapes):
    srcs = []
    sidx = eidx = 0
    for shape in spatial_shapes:
      h, w = shape
      eidx = sidx + h * w
      srcs += [rearrange(features[:, sidx:eidx], 'b (h w) c -> b c h w', h=h)]
      sidx = eidx
    return srcs


  def decode_mot_cls_box(self, target_feat, init_reference, inter_references):
    outputs_classes = []
    outputs_coords = []
    for lvl in range(target_feat.shape[0]):
      if lvl == 0:
        reference = init_reference
      else:
        reference = inter_references[lvl - 1]
      reference = inverse_sigmoid(reference)
      outputs_class = self.class_head[lvl](target_feat[lvl])
      tmp = self.box_head[lvl](target_feat[lvl])
      if reference.shape[-1] == 4:
        tmp += reference
      else:
        assert reference.shape[-1] == 2
        tmp[..., :2] += reference
      outputs_coord = tmp.sigmoid()
      outputs_classes.append(outputs_class)
      outputs_coords.append(outputs_coord)
    outputs_class = torch.stack(outputs_classes)
    outputs_coord = torch.stack(outputs_coords)
    return outputs_class, outputs_coord

  @torch.jit.unused
  def _set_aux_loss(self, outputs_class, outputs_coord):
    # this is a workaround to make torchscript happy, as torchscript
    # doesn't support dictionary with non-homogeneous values, such
    # as a dict having both a Tensor and a list.
    return [{
        'pred_logits': a,
        'pred_boxes': b
    } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

  @property
  def device(self):
    return self.pixel_mean.device

  def track(self, feature, mask, pre_embed=None, targets=None):
    if pre_embed is not None:
      pre_feat = pre_embed['feat']
    else:
      pre_feat = feature
    srcs, masks, posembs = self.get_mot_features(feature, mask)

    ### flatten feats for attention
    spatial_shapes = torch.tensor([list(src.shape[-2:]) for src in srcs],
                                  device=srcs[0].device,
                                  dtype=torch.long)
    valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
    srcs, masks, posembs = self.flatten_feats(srcs, masks, posembs)
    memory = self.encoder(srcs, spatial_shapes, valid_ratios, posembs, masks)

    ### detection
    target_emb, init_ref, inter_ref = self.decode_targets(
        memory, masks, spatial_shapes, valid_ratios)
    outputs_class, outputs_coord = self.decode_mot_cls_box(
        target_emb, init_ref, inter_ref)
    cur_hs = target_emb

    cur_class = outputs_class[-1]
    cur_box = outputs_coord[-1]
    cur_reference = cur_box
    cur_tgt = cur_hs[-1]

    track_class = cur_class
    track_center = None
    search_feat = self.unflatten_features(memory, spatial_shapes)
    if pre_embed is not None:
      # track mode
      pre_reference, pre_tgt = pre_embed['reference'], pre_embed['tgt']
      proposals = self.get_proposals(pre_reference, spatial_shapes, valid_ratios)
      track_box, track_center, track_embs = self.track_head(search_feat, pre_tgt, proposals)
      track_box = self.postprocess_boxes(track_box[-1:], spatial_shapes,
                                         valid_ratios)[0]
      track_center = track_center[-1] if len(track_center) > 1 else None
    else:
      proposals = self.get_proposals(cur_box, spatial_shapes, valid_ratios)
      _, _, track_embs = self.track_head(search_feat, cur_tgt, proposals)
      track_box = cur_box

    out = {
        'pred_logits': cur_class,
        'pred_boxes': cur_box,
        'detection_embs': target_emb[-1],
        'tracking_logits': track_class,
        'tracking_boxes': track_box,
        'tracking_embs': track_embs[-1],
        'tracking_centers': track_center
    }

    pre_embed = {'reference': cur_reference, 'tgt': cur_tgt, 'feat': feature}
    return out, pre_embed
