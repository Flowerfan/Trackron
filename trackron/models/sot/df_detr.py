import math
import torch
import torch.nn as nn
from typing import List, Optional
from einops import rearrange, repeat
from torchvision.ops import RoIAlign, box_convert
from torch.nn.init import normal_

from trackron.config import configurable
from trackron.models.layers.position_embedding import build_position_encoding
from trackron.models.box_heads import build_box_head
from trackron.structures import NestedTensor
from trackron.models.extractor import DeformableTransformerEncoderLayer, DeformableTransformerEncoder, DeformableTransformerDecoderLayer, DeformableTransformerDecoder

from trackron.external import MSDeformAttn
from .build import SOT_HEAD_REGISTRY


@SOT_HEAD_REGISTRY.register()
class DeformableTransformerHead(nn.Module):

  @configurable
  def __init__(
      self,
      *,
      feature_layers: List[str],
      input_proj: nn.Module,
      combine: nn.Module,
      pos_emb: nn.Module,
      encoder: nn.Module,
      decoder: nn.Module,
      box_head: nn.Module,
      refine_head: nn.Module,
      roi_aligns: nn.Module,
      level_embed: Optional[nn.Module] = None,
      two_stage: bool = False):
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
    self.input_proj = input_proj
    self.combine = combine
    self.pos_emb = pos_emb
    self.encoder = encoder
    self.decoder = decoder
    self.box_head = box_head
    self.refine_head = refine_head
    self.roi_aligns = roi_aligns
    self.level_embed = level_embed
    self.two_stage = two_stage
    self._reset_parameters()

  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    hidden_dim = cfg.FEATURE_DIM
    pos_emb = build_position_encoding(cfg.POSITION_EMBEDDING, hidden_dim)
    feature_layers = cfg.FEATURE_LAYERS
    num_feature_layers = cfg.NUM_FEATURE_LAYERS
    two_stage = cfg.TWO_STAGE
    out_channels = [in_feature_channels[layer] for layer in feature_layers]
    input_proj = [
        nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        ) for in_channels in out_channels
    ]
    roi_aligns = [
        RoIAlign(1, 1 / 2**(int(layer[-1]) + 1), 2, aligned=True)
        for layer in feature_layers
    ]
    if num_feature_layers > len(feature_layers):
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
      roi_aligns += [RoIAlign(1, 1 / 2**(int(feature_layers[-1][-1]) + 2), 2, aligned=True)]

    input_proj = nn.ModuleList(input_proj)
    roi_aligns = nn.ModuleList(roi_aligns)
    query_emb = nn.Embedding(1, hidden_dim) if cfg.USE_QUERY_EMB else None
    combine = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

    target_proj = nn.Sequential(
        nn.Linear(hidden_dim * len(input_proj), hidden_dim),
        nn.LayerNorm(hidden_dim))
    return_intermediate_dec = False

    encoder_layer = DeformableTransformerEncoderLayer(
        hidden_dim, cfg.ENCODER.DIM_FEEDFORWARD, cfg.ENCODER.DROPOUT,
        cfg.ENCODER.NORM, num_feature_layers, cfg.ENCODER.HEADS)
    encoder = DeformableTransformerEncoder(encoder_layer,
                                                cfg.ENCODER.NUM_LAYERS)
    decoder_layer = DeformableTransformerDecoderLayer(
        hidden_dim, cfg.DECODER.DIM_FEEDFORWARD, cfg.DECODER.DROPOUT,
        cfg.DECODER.NORM, num_feature_layers, cfg.DECODER.HEADS)
    decoder = DeformableTransformerDecoder(decoder_layer,
                                                cfg.DECODER.NUM_LAYERS,
                                                return_intermediate_dec)
    level_embed = nn.Parameter(torch.Tensor(num_feature_layers,
                                                 hidden_dim))

    refine_head = None
    if cfg.BOX_REFINE:
      refine_head = build_box_head(cfg.REFINE_HEAD)
    return {
        "pos_emb": pos_emb,
        "feature_layers": feature_layers,
        "input_proj": input_proj,
        "combine": combine,
        "encoder": encoder,
        "decoder": decoder,
        "box_head": build_box_head(cfg.BOX_HEAD),
        "refine_head": refine_head,
        "roi_aligns": roi_aligns,
        "level_embed": level_embed,
        "two_stage": two_stage,
    }

  def _reset_parameters(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    for m in self.modules():
      if isinstance(m, MSDeformAttn):
        m._reset_parameters()
    normal_(self.level_embed)

  def forward(self,
              template_features,
              template_boxes,
              search_features,
              search_boxes,
              template_masks=None,
              search_masks=None):
    # if not self.training:
    # def forward(self, template_imgs, search_imgs, template_bb, search_proposals, *args, **kwargs):
    #### process feat
    temp_srcs, temp_masks, temp_posembs = self.get_process_features(template_features, template_features, template_masks)
    search_srcs, search_masks, search_posembs = self.get_process_features(search_features, template_features, search_masks)

    ### flatten feats for attention
    spatial_shapes = torch.tensor([list(src.shape[-2:]) for src in temp_srcs],
                                  device=template_boxes.device,
                                  dtype=torch.long)
    valid_ratios = torch.stack([self.get_valid_ratio(m) for m in temp_masks], 1)
    temp_srcs, temp_masks, temp_posembs = self.flatten_feats(
        temp_srcs, temp_masks, temp_posembs)
    search_srcs, search_masks, search_posembs = self.flatten_feats(
        search_srcs, search_masks, search_posembs)

    # encoder
    temp_memory = self.encoder(temp_srcs, spatial_shapes, valid_ratios, temp_posembs, temp_masks)
    search_memory = self.encoder(search_srcs, spatial_shapes, valid_ratios, search_posembs, search_masks)

    tgt = self.extract_roi_target(temp_memory, template_boxes, spatial_shapes)
    reference = box_convert(template_boxes, 'xyxy', 'cxcywh') / 352
    reference = repeat(reference, 'b c -> b l c', l=len(spatial_shapes)).detach()
    reference = torch.full_like(reference, 0.5)
    query_embed = None

    # decoder
    temp_target, _ = self.decoder(tgt, reference, temp_memory, spatial_shapes,
                               valid_ratios, query_embed, temp_masks)
    search_target, _ = self.decoder(tgt, reference, search_memory, spatial_shapes,
                               valid_ratios, query_embed, search_masks)
    memory = torch.stack([temp_memory, search_memory], dim=1).flatten(0,1)
    target_feat = torch.stack([temp_target, search_target], dim=1).flatten(0,1)
    # Run the box decoder module
    # pred_boxes = self.decode_box(search_memory, search_target, spatial_shapes, layer=0)
    pred_boxes = self.decode_box(memory, target_feat, spatial_shapes, layer=0)
    return {"pred_boxes": pred_boxes}

  def refine_boxes(self, search_feat, target_feat, pred_boxes):
    """[refine boxes]

    Args:
        search_feat ([type]): [K B C]
        target_feat ([type]): [N B C]
        pred_boxes ([type]): [N B C]
    """
    K, B, C = search_feat.shape
    sz = int(math.sqrt(K))
    search_feat = rearrange(search_feat, '(H W) B C -> B C H W', H=sz, W=sz)
    proposals = rearrange(pred_boxes, 'N B C -> B N C') * sz * 16
    target_feat = rearrange(target_feat, 'N B C -> B N C')
    refine_proposals , _ = self.refine_head(search_feat, target_feat, proposals)
    refine_proposals = [rearrange(boxes, 'B N C -> N B C') / sz / 16 for boxes in [proposals] + refine_proposals]
    return refine_proposals

  def get_valid_ratio(self, mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio

  def decode_box(self, memory, target_feat, spatial_shapes, layer=1, training=True):
    srcs = []
    sidx = 0
    memory = rearrange(memory, 'b k c -> k b c')
    target_feat = rearrange(target_feat, 'b n c -> n b c')
    for shape in spatial_shapes:
      eidx = sidx + shape[-1] * shape[-2]
      srcs += [memory[sidx:eidx]]
      sidx = eidx
    pred_boxes = self.box_head(target_feat, srcs[layer])
    if self.refine_head is not None:
      pred_boxes = self.refine_boxes(srcs[layer], target_feat, pred_boxes)
      pred_boxes = [boxes.mean(1) for boxes in pred_boxes]
      if training:
        return pred_boxes
      return pred_boxes[-1]
    return pred_boxes.mean(1)

  def get_feature_maskposemb(self, feat, mask):
    B, C, H, W = feat.shape
    mask = torch.zeros((B, H, W), dtype=torch.bool, device=feat.device)
    # mask = F.interpolate(mask.float(), size=(H, W)).to(torch.bool).flatten(0, 1)
    # assert not mask.flatten(-2).all(-1).any(), print(
    #     'mask wrong, will get nan loss nan')
    nested_feat = NestedTensor(feat, mask)
    posemb = self.pos_emb(nested_feat).to(feat.dtype)
    return mask, posemb

  def get_process_features(self, feat1, feat2, feat_mask):
    srcs = []
    masks = []
    posembs = []
    for idx, layer in enumerate(self.feature_layers):
      layer_feat1 = self.input_proj[idx](feat1[layer])
      layer_feat2 = self.input_proj[idx](feat2[layer])
      feat = self.combine(torch.cat([layer_feat1, layer_feat2], dim=1))
      feat = layer_feat1
      mask, posemb = self.get_feature_maskposemb(feat, feat_mask[None])
      srcs += [feat]
      masks += [mask]
      posembs += [posemb]

    if len(self.feature_layers) < len(self.input_proj):
      srcs += [
          self.combine(
              torch.cat([
                  self.input_proj[-1](feat1[layer]), self.input_proj[-1](
                      feat2[layer])
              ],
                        dim=1))
      ]
      mask, posemb = self.get_feature_maskposemb(srcs[-1], feat_mask[None])
      masks += [mask]
      posembs += [posemb]
    return srcs, masks, posembs

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

  def extract_roi_target(self, flatten_feats, bboxes, spatial_shapes):
    bboxes = list(torch.split(bboxes, 1))
    target_feat = []
    sidx = 0
    for idx, shape in enumerate(spatial_shapes):
      h, w = shape[-2:]
      eidx = sidx + h * w
      feat = rearrange(flatten_feats[:, sidx:eidx, :],
                       'b (h w) c -> b c h w',
                       h=h,
                       w=w)
      target_feat += [self.roi_aligns[idx](feat, bboxes).flatten(-3)]
      sidx = eidx
    target_feat = torch.stack(target_feat, dim=1)
    return target_feat

  def track_init(self, features, boxes, masks=None):
    srcs, masks, posembs = self.get_process_features(features, features, masks)
    ### flatten feats for attention
    self.spatial_shapes = torch.tensor([list(src.shape[-2:]) for src in srcs],
                                  device=boxes.device,
                                  dtype=torch.long)
    self.valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
    srcs, masks, posembs = self.flatten_feats(srcs, masks, posembs)

    # encoder
    memory = self.encoder(srcs, self.spatial_shapes, self.valid_ratios, posembs, masks)
    target_features = self.extract_roi_target(memory, boxes, self.spatial_shapes)
    reference = box_convert(boxes, 'xyxy', 'cxcywh') / 384
    reference = repeat(reference, 'b c -> b l c', l=len(self.spatial_shapes))
    reference = torch.full_like(reference, 0.5)
    results = {
        "template_features": features,
        "target_features": target_features,
        "reference": reference
    }
    return results


  def track(self, features, masks, ref_info):
    template_features = ref_info['template_features']
    tgt, reference = ref_info['target_features'], ref_info[
        'reference']
    srcs, masks, posembs = self.get_process_features(features, template_features, masks)
    srcs, masks, posembs = self.flatten_feats(srcs, masks, posembs)
    memory = self.encoder(srcs, self.spatial_shapes, self.valid_ratios, posembs, masks)
    # decoder
    query_embed = None
    target_feat, reference = self.decoder(tgt, reference, srcs, self.spatial_shapes,
                               self.valid_ratios, query_embed, masks)
    # Run the box decoder module
    pred_boxes = self.decode_box(memory, target_feat, self.spatial_shapes, layer=0, training=False)
    results = {'pred_boxes': pred_boxes.reshape(-1,4).mean(0)}
    return results
