# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import math

import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from trackron.external import MSDeformAttn
from .build import EXTRACTOR_REGISTRY
from .encoder import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
from .decoder import DeformableTransformerDecoder, DeformableTransformerDecoderLayer




@EXTRACTOR_REGISTRY.register()
class DeformableTransformer(nn.Module):

  def __init__(self,
               d_model=256,
               nhead=8,
               num_encoder_layers=6,
               num_decoder_layers=6,
               dim_feedforward=1024,
               dropout=0.1,
               activation="relu",
               return_intermediate_dec=False,
               num_feature_levels=4,
               dec_n_points=4,
               enc_n_points=4,
               two_stage=False,
               two_stage_num_proposals=300,
               checkpoint_enc_ffn=False,
               checkpoint_dec_ffn=False):
    super().__init__()

    self.d_model = d_model
    self.nhead = nhead
    self.two_stage = two_stage
    self.two_stage_num_proposals = two_stage_num_proposals

    encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                      dropout, activation,
                                                      num_feature_levels, nhead,
                                                      enc_n_points,
                                                      checkpoint_enc_ffn)
    self.encoder = DeformableTransformerEncoder(encoder_layer,
                                                num_encoder_layers)
    decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                      dropout, activation,
                                                      num_feature_levels, nhead,
                                                      dec_n_points,
                                                      checkpoint_dec_ffn)
    self.decoder = DeformableTransformerDecoder(decoder_layer,
                                                num_decoder_layers,
                                                return_intermediate_dec)
    self.decoder_track = DeformableTransformerDecoder(decoder_layer,
                                                      num_decoder_layers,
                                                      return_intermediate_dec)

    self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

    if two_stage:
      self.enc_output = nn.Linear(d_model, d_model)
      self.enc_output_norm = nn.LayerNorm(d_model)
      self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
      self.pos_trans_norm = nn.LayerNorm(d_model * 2)
    else:
      self.reference_points = nn.Linear(d_model, 2)

    self._reset_parameters()

  def _reset_parameters(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    for m in self.modules():
      if isinstance(m, MSDeformAttn):
        m._reset_parameters()
    if not self.two_stage:
      xavier_uniform_(self.reference_points.weight.data, gain=1.0)
      constant_(self.reference_points.bias.data, 0.)
    normal_(self.level_embed)

  def get_proposal_pos_embed(self, proposals):
    num_pos_feats = 128
    temperature = 10000
    scale = 2 * math.pi

    dim_t = torch.arange(num_pos_feats,
                         dtype=torch.float32,
                         device=proposals.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
    # N, L, 4
    proposals = proposals.sigmoid() * scale
    # N, L, 4, 128
    pos = proposals[:, :, :, None] / dim_t
    # N, L, 4, 64, 2
    pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                      dim=4).flatten(2)
    return pos

  def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                   spatial_shapes):
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
      mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(
          N_, H_, W_, 1)
      valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
      valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

      grid_y, grid_x = torch.meshgrid(
          torch.linspace(0,
                         H_ - 1,
                         H_,
                         dtype=torch.float32,
                         device=memory.device),
          torch.linspace(0,
                         W_ - 1,
                         W_,
                         dtype=torch.float32,
                         device=memory.device))
      grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

      scale = torch.cat(
          [valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
      grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
      wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
      proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
      proposals.append(proposal)
      _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) &
                              (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid,
                                                    float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1),
                                              float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    output_memory = self.enc_output_norm(self.enc_output(output_memory))
    return output_memory, output_proposals

  def get_valid_ratio(self, mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio

  def forward(self,
              srcs,
              masks,
              pos_embeds,
              query_embed=None,
              pre_reference=None,
              pre_tgt=None,
              memory=None):
    # prepare input for encoder
    src_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
      bs, c, h, w = src.shape
      spatial_shape = (h, w)
      spatial_shapes.append(spatial_shape)
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
    spatial_shapes = torch.as_tensor(spatial_shapes,
                                     dtype=torch.long,
                                     device=src_flatten.device)
    valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

    # encoder
    if memory is None:
      memory = self.encoder(src_flatten, spatial_shapes, valid_ratios,
                            lvl_pos_embed_flatten, mask_flatten)

    # prepare input for decoder
    bs, _, c = memory.shape
    if pre_reference is not None:
      if self.two_stage and self.training:
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = self.decoder.bbox_embed[
            self.decoder.num_layers](output_memory) + output_proposals

      tgt = pre_tgt
      reference_points = pre_reference
      init_reference_out = reference_points

      query_embed = None
      # decoder
      hs, inter_references = self.decoder_track(tgt, reference_points, memory,
                                                spatial_shapes, valid_ratios,
                                                query_embed, mask_flatten)
      inter_references_out = inter_references
    else:
      if self.two_stage:
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = self.decoder.bbox_embed[
            self.decoder.num_layers](output_memory) + output_proposals

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        init_reference_out = reference_points
        pos_trans_out = self.pos_trans_norm(
            self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
      else:
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points
      query_embed = None
      # decoder
      hs, inter_references = self.decoder(tgt, reference_points, memory,
                                          spatial_shapes, valid_ratios,
                                          query_embed, mask_flatten)
      inter_references_out = inter_references

    if self.two_stage and self.training:
      return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact.sigmoid(
      ), memory

    return hs, init_reference_out, inter_references_out, None, None, memory

