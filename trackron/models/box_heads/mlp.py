import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, reduce
from timm.models.layers import DropPath
from trackron.config import configurable

from .build import BOX_HEAD_REGISTRY


@BOX_HEAD_REGISTRY.register()
class MLP(nn.Module):
  """ Very simple multi-layer perceptron (also called FFN)"""

  @configurable
  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
    super().__init__()
    self.num_layers = num_layers
    h = [hidden_dim] * (num_layers - 1)
    self.layers = nn.ModuleList(
        nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

  @classmethod
  def from_config(cls, cfg):
    return {
        "input_dim": cfg.INPUT_DIM,
        "hidden_dim": cfg.HIDDEN_DIM,
        "output_dim": cfg.OUTPUT_DIM,
        "num_layers": cfg.NUM_LAYERS
    }

  def forward(self, x):
    for i, layer in enumerate(self.layers):
      x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
    return x


class PreNormResidual(nn.Module):

  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.LayerNorm(dim)

  def forward(self, x):
    return self.fn(self.norm(x)) + x


def FeedForward(input_dim, hidden_dim, dropout=0., dense=nn.Linear):
  return nn.Sequential(dense(input_dim, input_dim // 4), nn.GELU(),
                       DropPath(dropout), dense(input_dim // 4, input_dim),
                       DropPath(dropout))


@BOX_HEAD_REGISTRY.register()
class MLPMixer(nn.Module):

  @configurable
  def __init__(self, input_dim, hidden_dim, patch_dim, num_layers, dropout=0.):
    super().__init__()
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    self.tl_layers = nn.Sequential(
        *[
            nn.Sequential(
                PreNormResidual(
                    input_dim,
                    FeedForward(patch_dim, hidden_dim, dropout, chan_first)),
                PreNormResidual(
                    input_dim,
                    FeedForward(input_dim, hidden_dim, dropout, chan_last)))
            for _ in range(num_layers)
        ], nn.Linear(input_dim, 1))
    self.br_layers = nn.Sequential(
        *[
            nn.Sequential(
                PreNormResidual(
                    input_dim,
                    FeedForward(patch_dim, hidden_dim, dropout, chan_first)),
                PreNormResidual(
                    input_dim,
                    FeedForward(input_dim, hidden_dim, dropout, chan_last)))
            for _ in range(num_layers)
        ], nn.Linear(input_dim, 1))

  @classmethod
  def from_config(cls, cfg):
    return {
        "input_dim": cfg.INPUT_DIM,
        "hidden_dim": cfg.HIDDEN_DIM,
        "patch_dim": cfg.PATCH_DIM,
        "num_layers": cfg.NUM_LAYERS
    }

  def forward(self, x):
    tl_score_map = self.tl_layers(x)
    br_score_map = self.br_layers(x)
    return tl_score_map.squeeze(-1), br_score_map.squeeze(-1)


class Affine(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.g = nn.Parameter(torch.ones(1, 1, dim))
    self.b = nn.Parameter(torch.zeros(1, 1, dim))

  def forward(self, x):
    return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):  # https://arxiv.org/abs/2103.17239

  def __init__(self, dim, fn, init_eps=1e-5):
    super().__init__()

    scale = torch.zeros(1, 1, dim).fill_(init_eps)
    self.scale = nn.Parameter(scale)
    self.affine = Affine(dim)
    self.fn = fn

  def forward(self, x):
    return self.fn(self.affine(x)) * self.scale + x


@BOX_HEAD_REGISTRY.register()
class MLPRes(nn.Module):

  @configurable
  def __init__(self, input_dim, hidden_dim, patch_dim, num_layers, dropout=0.):
    super().__init__()
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
    wrapper = lambda i, fn: PreAffinePostLayerScale(input_dim, fn)
    self.tl_layers = nn.Sequential(
        *[
            nn.Sequential(
                wrapper(i, nn.Conv1d(patch_dim, patch_dim, 1)),
                wrapper(
                    i,
                    nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, input_dim))))
            for i in range(num_layers)
        ], Affine(input_dim), nn.Linear(input_dim, 1))
    self.br_layers = nn.Sequential(
        *[
            nn.Sequential(
                wrapper(i, nn.Conv1d(patch_dim, patch_dim, 1)),
                wrapper(
                    i,
                    nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, input_dim))))
            for i in range(num_layers)
        ], Affine(input_dim), nn.Linear(input_dim, 1))

  @classmethod
  def from_config(cls, cfg):
    return {
        "input_dim": cfg.INPUT_DIM,
        "hidden_dim": cfg.HIDDEN_DIM,
        "patch_dim": cfg.PATCH_DIM,
        "num_layers": cfg.NUM_LAYERS
    }

  def forward(self, x):
    tl_score_map = self.tl_layers(x)
    br_score_map = self.br_layers(x)
    return tl_score_map.squeeze(-1), br_score_map.squeeze(-1)
