from typing import Optional

import torch
from timm.models.layers import DropPath
from torch import Tensor, nn
from trackron.external import MSDeformAttn
from trackron.utils.misc import (clone_modules, get_activation_fn, inverse_sigmoid)

from .build import EXTRACTOR_REGISTRY


@EXTRACTOR_REGISTRY.register()
class TransformerDecoder(nn.Module):

  def __init__(self,
               decoder_layer,
               num_layers,
               norm=None,
               return_intermediate=False):
    super().__init__()
    self.layers = clone_modules(decoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = norm
    self.return_intermediate = return_intermediate

  def forward(self,
              tgt,
              memory,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None,
              memory_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None,
              query_pos: Optional[Tensor] = None):
    output = tgt

    intermediate = []

    for layer in self.layers:
      output = layer(output,
                     memory,
                     tgt_mask=tgt_mask,
                     memory_mask=memory_mask,
                     tgt_key_padding_mask=tgt_key_padding_mask,
                     memory_key_padding_mask=memory_key_padding_mask,
                     pos=pos,
                     query_pos=query_pos)
      if self.return_intermediate:
        intermediate.append(self.norm(output))

    if self.norm is not None:
      output = self.norm(output)
      if self.return_intermediate:
        intermediate.pop()
        intermediate.append(output)

    if self.return_intermediate:
      return torch.stack(intermediate)

    return output
    # return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

  def __init__(self,
               d_model,
               nhead,
               dim_feedforward=2048,
               dropout=0.1,
               drop_func=nn.Dropout,
               activation="relu",
               normalize_before=False,
               divide_norm=False):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = DropPath(dropout) if dropout > 0. else nn.Identity()
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout1 = drop_func(dropout) if dropout > 0. else nn.Identity()
    self.dropout2 = drop_func(dropout) if dropout > 0. else nn.Identity()
    self.dropout3 = drop_func(dropout) if dropout > 0. else nn.Identity()

    self.activation = get_activation_fn(activation)
    self.normalize_before = normalize_before

    self.divide_norm = divide_norm
    self.scale_factor = float(d_model // nhead)**0.5

  def with_pos_embed(self, tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

  def forward_post(self,
                   tgt,
                   memory,
                   tgt_mask: Optional[Tensor] = None,
                   memory_mask: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   pos: Optional[Tensor] = None,
                   query_pos: Optional[Tensor] = None):
    # self-attention
    q = k = self.with_pos_embed(
        tgt, query_pos)  # Add object query to the query and key
    if self.divide_norm:
      q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
      k = k / torch.norm(k, dim=-1, keepdim=True)
    tgt2 = self.self_attn(q,
                          k,
                          value=tgt,
                          attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    # mutual attention
    queries, keys = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(
        memory, pos)
    if self.divide_norm:
      queries = queries / torch.norm(queries, dim=-1,
                                     keepdim=True) * self.scale_factor
      keys = keys / torch.norm(keys, dim=-1, keepdim=True)
    tgt2 = self.multihead_attn(query=queries,
                               key=keys,
                               value=memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)
    return tgt

  def forward_pre(self,
                  tgt,
                  memory,
                  tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None,
                  tgt_key_padding_mask: Optional[Tensor] = None,
                  memory_key_padding_mask: Optional[Tensor] = None,
                  pos: Optional[Tensor] = None,
                  query_pos: Optional[Tensor] = None):
    tgt2 = self.norm1(tgt)
    q = k = self.with_pos_embed(tgt2, query_pos)
    tgt2 = self.self_attn(q,
                          k,
                          value=tgt2,
                          attn_mask=tgt_mask,
                          key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt2 = self.norm2(tgt)
    tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt2 = self.norm3(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
    tgt = tgt + self.dropout3(tgt2)
    return tgt

  def forward(self,
              tgt,
              memory,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None,
              memory_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None,
              query_pos: Optional[Tensor] = None):
    if self.normalize_before:
      return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                              tgt_key_padding_mask, memory_key_padding_mask,
                              pos, query_pos)
    return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                             tgt_key_padding_mask, memory_key_padding_mask, pos,
                             query_pos)


class TransformerSingleDecoderLayer(nn.Module):

  def __init__(self,
               d_model,
               nhead,
               dim_feedforward=2048,
               dropout=0.1,
               drop_func=nn.Dropout,
               activation="relu",
               normalize_before=False,
               divide_norm=False):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = drop_func(dropout) if dropout > 0. else nn.Identity()
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = drop_func(dropout) if dropout > 0. else nn.Identity()
    self.dropout2 = drop_func(dropout) if dropout > 0. else nn.Identity()

    self.activation = get_activation_fn(activation)
    self.normalize_before = normalize_before

    self.divide_norm = divide_norm
    self.scale_factor = float(d_model // nhead)**0.5

  def with_pos_embed(self, tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

  def forward_post(self,
                   tgt,
                   memory,
                   tgt_mask: Optional[Tensor] = None,
                   memory_mask: Optional[Tensor] = None,
                   tgt_key_padding_mask: Optional[Tensor] = None,
                   memory_key_padding_mask: Optional[Tensor] = None,
                   pos: Optional[Tensor] = None,
                   query_pos: Optional[Tensor] = None):
    # mutual attention
    queries, keys = self.with_pos_embed(tgt, query_pos), self.with_pos_embed(
        memory, pos)
    if self.divide_norm:
      queries = queries / torch.norm(queries, dim=-1,
                                     keepdim=True) * self.scale_factor
      keys = keys / torch.norm(keys, dim=-1, keepdim=True)
    tgt2 = self.multihead_attn(query=queries,
                               key=keys,
                               value=memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)
    return tgt

  def forward_pre(self,
                  tgt,
                  memory,
                  tgt_mask: Optional[Tensor] = None,
                  memory_mask: Optional[Tensor] = None,
                  tgt_key_padding_mask: Optional[Tensor] = None,
                  memory_key_padding_mask: Optional[Tensor] = None,
                  pos: Optional[Tensor] = None,
                  query_pos: Optional[Tensor] = None):
    tgt2 = self.norm1(tgt)
    tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                               key=self.with_pos_embed(memory, pos),
                               value=memory,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt2 = self.norm2(tgt)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
    tgt = tgt + self.dropout2(tgt2)
    return tgt

  def forward(self,
              tgt,
              memory,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              tgt_key_padding_mask: Optional[Tensor] = None,
              memory_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None,
              query_pos: Optional[Tensor] = None):
    if self.normalize_before:
      return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                              tgt_key_padding_mask, memory_key_padding_mask,
                              pos, query_pos)
    return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                             tgt_key_padding_mask, memory_key_padding_mask, pos,
                             query_pos)


class DeformableTransformerDecoderLayer(nn.Module):

  def __init__(self,
               d_model=256,
               d_ffn=1024,
               dropout=0.1,
               activation="relu",
               n_levels=4,
               n_heads=8,
               n_points=4,
               checkpoint_ffn=False):
    super().__init__()

    # cross attention
    self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model)

    # self attention
    self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(d_model)

    # ffn
    self.linear1 = nn.Linear(d_model, d_ffn)
    self.activation = get_activation_fn(activation)
    self.dropout3 = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ffn, d_model)
    self.dropout4 = nn.Dropout(dropout)
    self.norm3 = nn.LayerNorm(d_model)

    # use torch.utils.checkpoint.checkpoint to save memory
    self.checkpoint_ffn = checkpoint_ffn

  @staticmethod
  def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos

  def forward_ffn(self, tgt):
    tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout4(tgt2)
    tgt = self.norm3(tgt)
    return tgt

  def forward(self,
              tgt,
              query_pos,
              reference_points,
              src,
              src_spatial_shapes,
              src_padding_mask=None):
    # self attention
    q = k = self.with_pos_embed(tgt, query_pos)
    tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                          tgt.transpose(0, 1))[0].transpose(0, 1)
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)

    # cross attention
    tgt2 = self.cross_attn(self.with_pos_embed(tgt,
                                               query_pos), reference_points,
                           src, src_spatial_shapes, src_padding_mask)
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)

    # ffn
    if self.checkpoint_ffn:
      tgt = torch.utils.checkpoint.checkpoint(self.forward_ffn, tgt)
    else:
      tgt = self.forward_ffn(tgt)

    return tgt

class DeformableTransformerDecoder(nn.Module):

  def __init__(self, decoder_layer, num_layers, return_intermediate=False):
    super().__init__()
    self.layers = clone_modules(decoder_layer, num_layers)
    self.num_layers = num_layers
    self.return_intermediate = return_intermediate
    # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
    self.bbox_embed = None
    self.class_embed = None

  def forward(self,
              tgt,
              reference_points,
              src,
              src_spatial_shapes,
              src_valid_ratios,
              query_pos=None,
              src_padding_mask=None):
    output = tgt

    intermediate = []
    intermediate_reference_points = []
    for lid, layer in enumerate(self.layers):
      if reference_points.shape[-1] == 4:
        reference_points_input = reference_points[:, :, None] \
                                 * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
      else:
        assert reference_points.shape[-1] == 2
        reference_points_input = reference_points[:, :,
                                                  None] * src_valid_ratios[:,
                                                                           None]
      output = layer(output, query_pos, reference_points_input, src,
                     src_spatial_shapes, src_padding_mask)

      # hack implementation for iterative bounding box refinement
      if self.bbox_embed is not None:
        tmp = self.bbox_embed[lid](output)
        if reference_points.shape[-1] == 4:
          new_reference_points = tmp + inverse_sigmoid(reference_points)
          new_reference_points = new_reference_points.sigmoid()
        else:
          assert reference_points.shape[-1] == 2
          new_reference_points = tmp
          new_reference_points[
              ..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
          new_reference_points = new_reference_points.sigmoid()
        reference_points = new_reference_points.detach()

      if self.return_intermediate:
        intermediate.append(output)
        intermediate_reference_points.append(reference_points)

    if self.return_intermediate:
      return torch.stack(intermediate), torch.stack(
          intermediate_reference_points)

    return output, reference_points