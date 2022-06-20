import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from trackron.config import configurable
from trackron.models.extractor import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from trackron.models.layers.position_embedding import build_position_encoding
from einops import rearrange
from torchvision.ops import RoIAlign
from trackron.structures import NestedTensor

from .build import SOT_HEAD_REGISTRY
from trackron.models.box_heads import build_box_head


@SOT_HEAD_REGISTRY.register()
class TokenHead(nn.Module):

  @configurable
  def __init__(
      self,
      *,
      feature_layer: str,
      input_proj: nn.Module,
      pos_emb: nn.Module,
      encoder: nn.Module,
      decoder: nn.Module,
      box_head: nn.Module,
      refine_head: nn.Module,
      roi_align: nn.Module,
      target_proj: nn.Module,
      # box_head2: nn.Module,
      stride: Optional[int] = 16,
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
    self.feature_layer = feature_layer
    self.input_proj = input_proj
    self.pos_emb = pos_emb
    self.box_head = box_head
    self.refine_head = refine_head
    self.encoder = encoder
    self.decoder = decoder
    self.roi_align = roi_align
    self.query_embed = query_emb
    self.target_proj = target_proj

  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    hidden_dim = cfg.FEATURE_DIM
    pos_emb = build_position_encoding(cfg.POSITION_EMBEDDING, hidden_dim)
    in_cha = in_feature_channels[cfg.FEATURE_LAYER]
    kernel_sz = cfg.KERNEL_SZ
    input_proj = nn.Conv2d(in_cha, hidden_dim, kernel_size=kernel_sz)
    enc_layer = TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=cfg.ENCODER.HEADS,
        dim_feedforward=cfg.ENCODER.DIM_FEEDFORWARD,
        dropout=cfg.ENCODER.DROPOUT,
        activation=cfg.ENCODER.NORM,
        normalize_before=cfg.ENCODER.PRE_NORM)
    encoder = TransformerEncoder(
        encoder_layer=enc_layer,
        num_layers=cfg.ENCODER.NUM_LAYERS,
        norm=None)

    dec_layer = TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=cfg.DECODER.HEADS,
        dim_feedforward=cfg.DECODER.DIM_FEEDFORWARD,
        dropout=cfg.DECODER.DROPOUT,
        activation=cfg.DECODER.NORM,
        normalize_before=cfg.DECODER.PRE_NORM)
    decoder = TransformerDecoder(
        decoder_layer=dec_layer,
        num_layers=cfg.DECODER.NUM_LAYERS,
        norm=nn.LayerNorm(hidden_dim))

    query_emb = nn.Embedding(1, hidden_dim) if cfg.USE_QUERY_EMB else None
    stride = cfg.FEATURE_STRIDE
    roi_align = RoIAlign(kernel_sz, 1 / stride, 2, aligned=True)
    target_proj = nn.Linear(hidden_dim * kernel_sz ** 2, hidden_dim)
    refine_head = None
    if cfg.BOX_REFINE:
      refine_head = build_box_head(cfg.REFINE_HEAD)
    return {
        "pos_emb": pos_emb,
        "feature_layer": cfg.FEATURE_LAYER,
        "input_proj": input_proj,
        "encoder": encoder,
        "decoder": decoder,
        "box_head": build_box_head(cfg.BOX_HEAD),
        "refine_head": refine_head,
        "roi_align": roi_align,
        "query_emb": query_emb,
        "target_proj": target_proj
    }

  def forward(self,
              template_features,
              template_boxes,
              search_features,
              search_boxes,
              template_masks=None,
              search_masks=None):
    # if not self.training:
    # def forward(self, template_imgs, search_imgs, template_bb, search_proposals, *args, **kwargs):
    template_features = self.input_proj(template_features[self.feature_layer])
    template_masks, template_posembs = self.get_feature_maskposemb(
        template_features, template_masks)
    search_features = self.input_proj(search_features[self.feature_layer])
    search_masks, search_posembs = self.get_feature_maskposemb(
        search_features, search_masks)
    B, C, H, W = search_features.shape
    ### search feature spatial temporal encoding
    target_feature, target_mask, target_posemb = self.extract_target_feat(template_features, template_boxes, template_masks, template_posembs)

    target_feature = target_feature.repeat_interleave(2, dim=0)
    target_mask = target_mask.repeat_interleave(2, dim=0)
    target_posemb = target_posemb.repeat_interleave(2, dim=0)
    search_features = torch.stack([template_features, search_features], dim=1).flatten(0,1)
    search_masks = torch.stack([template_masks, search_masks], dim=1).flatten(0,1)
    search_posembs = torch.stack([template_posembs, search_posembs], dim=1).flatten(0,1)
    features, masks, posembs = self.fuse_feature(target_feature.flatten(-2), target_mask.flatten(-2), target_posemb.flatten(-2), search_features.flatten(-2), search_masks.flatten(-2), search_posembs.flatten(-2))

    features = self.encoder(features,
                            src_key_padding_mask=masks,
                            pos=posembs)
    target_feat = rearrange(features[:-H*W], 'n b c -> b (c n)')[None]
    search_feat = features[-H*W:]
    target_feat = self.target_proj(target_feat)

    # Run the box decoder module
    pred_boxes = self.box_head(target_feat, search_feat)
    if self.refine_head is not None:
      pred_boxes = self.refine_boxes(search_feat, target_feat, pred_boxes)
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
  
  def fuse_feature(self, target_features, target_masks, target_posembs, search_features, search_masks, search_posembs):
    features = torch.cat([target_features, search_features], dim=-1)
    masks = torch.cat([target_masks, search_masks], dim=-1)
    posembs = torch.cat([target_posembs, search_posembs], dim=-1)
    features = rearrange(features, 'b c k -> k b c')
    posembs = rearrange(posembs, 'b c k -> k b c')
    return features, masks, posembs

  def get_feature_maskposemb(self, feat, mask):
    if mask is None:
      return None, None
    mask = F.interpolate(mask[None].float(),
                         size=feat.shape[-2:]).to(torch.bool)[0]
    assert not mask.flatten(-2).all(-1).any(), print(
        'mask wrong, will get nan loss nan')
    nested_feat = NestedTensor(feat, mask)
    pos_emb = self.pos_emb(nested_feat).to(feat.dtype)
    return mask, pos_emb

  def extract_target_feat(self, feat, bb, mask=None, posemb=None):
    bb = [b.reshape(1, 4) for b in bb]
    target_feat = self.roi_align(feat, bb)
    # target_feat = self.target_proj(target_feat)
    assert mask is None or mask.dim(
    ) == 3, 'mask mut be None or dimension must be 3 (B H W)'
    target_mask = self.roi_align(
        mask.unsqueeze(1).float(),
        bb).squeeze(1).bool() if mask is not None else None
    target_posemb = self.roi_align(posemb,
                                    bb) if posemb is not None else None
    return target_feat, target_mask, target_posemb

  def track_init(self, features, boxes, masks=None):
    features = self.input_proj(features[self.feature_layer])
    masks, posembs = self.get_feature_maskposemb(features, masks)
    boxes = boxes.to(features.device)
    target_feature, target_mask, target_posemb = self.extract_target_feat(features, boxes, masks, posembs)

    return {"target_feature": target_feature.flatten(-2), "target_mask": target_mask.flatten(-2), "target_posemb": target_posemb.flatten(-2)}

  def track(self, features, masks, ref_info):
    cur_features = self.input_proj(features[self.feature_layer])
    B, C, H, W = cur_features.shape
    cur_masks, cur_posembs = self.get_feature_maskposemb(cur_features, masks)
    target_feature = ref_info['target_feature']
    target_mask = ref_info['target_mask']
    target_posemb = ref_info['target_posemb']
    features, masks, posembs = self.fuse_feature(target_feature, target_mask, target_posemb, cur_features.flatten(-2), cur_masks.flatten(-2), cur_posembs.flatten(-2))
    features = self.encoder(features,
                            src_key_padding_mask=masks,
                            pos=posembs)
    target_feat = rearrange(features[:-H*W], 'n b c -> b (c n)')[None]
    search_feat = features[-H*W:]
    target_feat = self.target_proj(target_feat)

    pred_boxes = self.box_head(target_feat, search_feat)
    if self.refine_head is not None:
      pred_boxes = self.refine_boxes(search_feat, target_feat, pred_boxes)[-1]
    boxes = pred_boxes.reshape(-1, 4).mean(0, keepdim=True) * 352
    cur_target_feature, cur_target_mask, cur_target_posemb = self.extract_target_feat(cur_features, boxes, cur_masks, cur_posembs)
    # if ref_info['target_feature'].shape[-1] < 2:
    #   ref_info['target_feature'] = torch.cat([target_feature, cur_target_feature.flatten(-2)], dim=-1)
    #   ref_info['target_mask'] = torch.cat([target_mask, cur_target_mask.flatten(-2)], dim=-1)
    #   ref_info['target_posemb'] = torch.cat([target_posemb, cur_target_posemb.flatten(-2)], dim=-1)
    # else:
    #   ref_info['target_feature'][..., -1:] = cur_target_feature.flatten(-2)
    #   ref_info['target_mask'][..., -1:] = cur_target_mask.flatten(-2)
    #   ref_info['target_posemb'][..., -1:] = cur_target_posemb.flatten(-2)
    results = {'pred_boxes': pred_boxes.reshape(-1,4).mean(0)}
    return results
