from typing import Optional

import torch
from torch import Tensor, nn
from trackron.external import MSDeformAttn
from trackron.utils.misc import (clone_modules, get_activation_fn)

from .build import EXTRACTOR_REGISTRY


@EXTRACTOR_REGISTRY.register()
class TransformerEncoder(nn.Module):

  def __init__(self, encoder_layer, num_layers, norm=None):
    super().__init__()
    self.layers = clone_modules(encoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = norm

  def forward(self,
              src,
              mask: Optional[Tensor] = None,
              src_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None):
    output = src

    for layer in self.layers:
      output = layer(output,
                     src_mask=mask,
                     src_key_padding_mask=src_key_padding_mask,
                     pos=pos)

    if self.norm is not None:
      output = self.norm(output)

    return output


class TransformerEncoderLayer(nn.Module):

  def __init__(self,
               d_model,
               nhead,
               dim_feedforward=2048,
               dropout=0.1,
               activation="relu",
               normalize_before=False,
               divide_norm=False,
               drop_func=nn.Dropout):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = drop_func(dropout) if dropout > 0. else nn.Identity()
    self.dropout = drop_func(dropout) if dropout > 0. else nn.Identity()
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = drop_func(dropout) if dropout > 0. else nn.Identity()
    self.dropout2 = drop_func(dropout) if dropout > 0. else nn.Identity()

    self.activation = get_activation_fn(activation)
    self.normalize_before = normalize_before  # first normalization, then add

    self.divide_norm = divide_norm
    self.scale_factor = float(d_model // nhead)**0.5

  def with_pos_embed(self, tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

  def forward_post(self,
                   src,
                   src_mask: Optional[Tensor] = None,
                   src_key_padding_mask: Optional[Tensor] = None,
                   pos: Optional[Tensor] = None):
    q = k = self.with_pos_embed(src, pos)  # add pos to src
    if self.divide_norm:
      # print("encoder divide by norm")
      q = q / torch.norm(q, dim=-1, keepdim=True) * self.scale_factor
      k = k / torch.norm(k, dim=-1, keepdim=True)
    src2 = self.self_attn(q,
                          k,
                          value=src,
                          attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src

  def forward_pre(self,
                  src,
                  src_mask: Optional[Tensor] = None,
                  src_key_padding_mask: Optional[Tensor] = None,
                  pos: Optional[Tensor] = None):
    src2 = self.norm1(src)
    q = k = self.with_pos_embed(src2, pos)
    src2 = self.self_attn(q,
                          k,
                          value=src2,
                          attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    src2 = self.norm2(src)
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
    src = src + self.dropout2(src2)
    return src

  def forward(self,
              src,
              src_mask: Optional[Tensor] = None,
              src_key_padding_mask: Optional[Tensor] = None,
              pos: Optional[Tensor] = None):
    if self.normalize_before:
      return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
    return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class DeformableTransformerEncoder(nn.Module):

  def __init__(self, encoder_layer, num_layers):
    super().__init__()
    self.layers = clone_modules(encoder_layer, num_layers)
    self.num_layers = num_layers

  @staticmethod
  def get_reference_points(spatial_shapes, valid_ratios, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):

      ref_y, ref_x = torch.meshgrid(
          torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
      ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
      ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
      ref = torch.stack((ref_x, ref_y), -1)
      reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points

  def forward(self,
              src,
              spatial_shapes,
              valid_ratios,
              pos=None,
              padding_mask=None):
    output = src
    reference_points = self.get_reference_points(spatial_shapes,
                                                 valid_ratios,
                                                 device=src.device)
    for layer in self.layers:
      output = layer(output, pos, reference_points, spatial_shapes,
                     padding_mask)

    return output


class DeformableTransformerEncoderLayer(nn.Module):

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

    # self attention
    self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
    self.dropout1 = nn.Dropout(dropout)
    self.norm1 = nn.LayerNorm(d_model)

    # ffn
    self.linear1 = nn.Linear(d_model, d_ffn)
    self.activation = get_activation_fn(activation)
    self.dropout2 = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ffn, d_model)
    self.dropout3 = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(d_model)

    # use torch.utils.checkpoint.checkpoint to save memory
    self.checkpoint_ffn = checkpoint_ffn

  @staticmethod
  def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos

  def forward_ffn(self, src):
    src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
    src = src + self.dropout3(src2)
    src = self.norm2(src)
    return src

  def forward(self,
              src,
              pos,
              reference_points,
              spatial_shapes,
              padding_mask=None):
    # self attention
    src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src,
                          spatial_shapes, padding_mask)
    src = src + self.dropout1(src2)
    src = self.norm1(src)

    # ffn
    if self.checkpoint_ffn:
      src = torch.utils.checkpoint.checkpoint(self.forward_ffn, src)
    else:
      src = self.forward_ffn(src)

    return src

