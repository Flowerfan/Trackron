from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from trackron.config import configurable
from trackron.data.utils import normalize_boxes
from trackron.models.box_heads import build_box_head
from trackron.models.extractor import (TransformerDecoder,
                                       TransformerDecoderLayer,
                                       TransformerEncoder,
                                       TransformerEncoderLayer)
from trackron.models.layers.position_embedding import build_position_encoding
from trackron.structures import NestedTensor

from .build import SOT_HEAD_REGISTRY


@SOT_HEAD_REGISTRY.register()
class STARK(nn.Module):

  @configurable
  def __init__(
      self,
      *,
      feature_layer: str,
      pos_emb: nn.Module,
      encoder: nn.Module,
      decoder: nn.Module,
      box_head: nn.Module,
      query_emb: nn.Module,
      input_proj: nn.Module,
      # box_head2: nn.Module,
      stride: Optional[int] = 16):
    """[summary]

    Args:
        feature_layer (list): [description]
        output_layers (list): [description]
        backbone (nn.Module): [description]
        cls_head (nn.Module): [description]
        box_head (nn.Module): [description]
        pixel_mean (Tuple[float]): [description]
        pixel_std (Tuple[float]): [description]
    """
    super().__init__()
    self.feature_layer = feature_layer
    self.pos_emb = pos_emb
    self.box_head = box_head
    # self.box_head2 = box_head2
    self.encoder = encoder
    self.decoder = decoder
    self.input_proj = input_proj
    self.stride = stride
    self.query_embed = query_emb


  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    hidden_dim = cfg.FEATURE_DIM
    pos_emb = build_position_encoding(cfg.POSITION_EMBEDDING, hidden_dim)
    enc_layer = TransformerEncoderLayer(
        d_model=cfg.ENCODER.HIDDEN_DIM,
        nhead=cfg.ENCODER.HEADS,
        dim_feedforward=cfg.ENCODER.DIM_FEEDFORWARD,
        dropout=cfg.ENCODER.DROPOUT,
        activation=cfg.ENCODER.NORM,
        normalize_before=cfg.ENCODER.PRE_NORM)
    encoder = TransformerEncoder(encoder_layer=enc_layer,
                                 num_layers=cfg.ENCODER.NUM_LAYERS,
                                 norm=None)

    dec_layer = TransformerDecoderLayer(
        d_model=cfg.DECODER.HIDDEN_DIM,
        nhead=cfg.DECODER.HEADS,
        dim_feedforward=cfg.DECODER.DIM_FEEDFORWARD,
        dropout=cfg.DECODER.DROPOUT,
        activation=cfg.DECODER.NORM,
        normalize_before=cfg.DECODER.PRE_NORM)
    decoder = TransformerDecoder(decoder_layer=dec_layer,
                                 num_layers=cfg.DECODER.NUM_LAYERS,
                                 norm=nn.LayerNorm(
                                     cfg.DECODER.HIDDEN_DIM))
    query_emb = nn.Embedding(1, hidden_dim)
    in_cha = in_feature_channels[cfg.FEATURE_LAYER]
    input_proj = nn.Conv2d(in_cha, hidden_dim, kernel_size=1)

    return {
        "pos_emb": pos_emb,
        "encoder": encoder,
        "decoder": decoder,
        "box_head": build_box_head(cfg.BOX_HEAD),
        "feature_layer": cfg.FEATURE_LAYER,
        "query_emb": query_emb,
        "input_proj": input_proj,
        # "input_format": cfg.INPUT.FORMAT,
        # "vis_period": cfg.VIS_PERIOD,
    }

  def forward(self,
              template_features,
              template_boxes,
              search_features,
              search_boxes,
              template_masks=None,
              search_masks=None):
    # Extract backbone features
    template_features = self.input_proj(template_features[self.feature_layer])
    template_mask, template_pos_emb = self.get_feature_maskposemb(
        template_features, template_masks[None])
    search_features = self.input_proj(search_features[self.feature_layer])
    search_mask, search_pos_emb = self.get_feature_maskposemb(
        search_features, search_masks[None])
    feat = torch.cat([template_features.flatten(-2),
                      search_features.flatten(-2)],
                     dim=-1)
    mask = torch.cat([template_mask.flatten(-2),
                      search_mask.flatten(-2)],
                     dim=-1)
    pos_emb = torch.cat(
        [template_pos_emb.flatten(-2),
         search_pos_emb.flatten(-2)], dim=-1)

    assert not search_mask.flatten(-2).all(-1).any(), print(
        'mask wrong, loss nan will result')

    ### search feature spatial temporal encoding
    search_box_pred = self.forward_transformer(feat, mask, pos_emb)

    return {'pred_boxes': search_box_pred}

  def forward_transformer(self, feat, mask, pos_emb):
    target_feat, feat = self.transformer(feat, mask=mask, pos_emb=pos_emb)
    search_feat = rearrange(feat[-400:], '(h w) b c -> b c h w', h=20, w=20)
    # Run the box decoder module
    target_box_pred = self.box_head(target_feat, search_feat)
    return target_box_pred

  def get_feature_maskposemb(self, feat, mask):
    C, H, W = feat.shape[1:]
    mask = F.interpolate(mask.float(), size=(H, W)).to(torch.bool).flatten(0, 1)
    nested_feat = NestedTensor(feat, mask)
    pos_emb = self.pos_emb(nested_feat).to(feat.dtype)
    return mask, pos_emb

  def transformer(self, feat, mask=None, pos_emb=None):
    B, C, HW = feat.shape
    feat = rearrange(feat, 'b c k -> k b c')
    if pos_emb is not None:
      pos_emb = rearrange(pos_emb, 'b c k -> k b c')
    feat = self.encoder(feat, src_key_padding_mask=mask, pos=pos_emb)
    ## decoder
    query_pos_emb = repeat(self.query_embed.weight, 'n c -> n b c', b=B)
    tgt = torch.zeros_like(query_pos_emb)
    target_feat = self.decoder(tgt,
                               feat,
                               memory_key_padding_mask=mask,
                               pos=pos_emb,
                               query_pos=query_pos_emb)
    return target_feat, feat

  def track(self, feature, mask=None, ref_info=None, init_box=None):
    cur_feature = self.input_proj(feature[self.feature_layer])
    cur_mask, cur_posemb = self.get_feature_maskposemb(cur_feature, mask[None])
    if ref_info is None:
      return {'target_feature': cur_feature, 'target_mask': cur_mask, 'target_posemb': cur_posemb}
      
    # B, C, H, W = cur_feature.shape
    target_feature = ref_info['target_feature']
    target_mask = ref_info['target_mask']
    target_posemb = ref_info['target_posemb']
    image_sizes = ref_info['image_sizes']
    feat = torch.cat([target_feature.flatten(-2),
                      cur_feature.flatten(-2)],
                     dim=-1)
    mask = torch.cat([target_mask.flatten(-2),
                      cur_mask.flatten(-2)],
                     dim=-1)
    pos_emb = torch.cat(
        [target_posemb.flatten(-2),
         cur_posemb.flatten(-2)], dim=-1)
    pred_boxes = self.forward_transformer(feat, mask, pos_emb)
    pred_boxes = normalize_boxes(
              pred_boxes,
              image_sizes,
              in_format='xyxy',
              out_format='xyxy', reverse=True) 
    results = {'pred_boxes': pred_boxes.reshape(-1,4).mean(0, keepdim=True)}
    return results

