import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .build import CLS_LOSS_REGISTRY, BOX_LOSS_REGISTRY



class MLRegression(nn.Module):
    """Maximum likelihood loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density=None, mc_dim=-1):
        """Args:
            scores: predicted score values. First sample must be ground-truth
            sample_density: probability density of the sample distribution
            gt_density: not used
            mc_dim: dimension of the MC samples. Only mc_dim=1 supported"""

        assert mc_dim == 1
        assert (sample_density[:,0,...] == -1).all()

        exp_val = scores[:, 1:, ...] - torch.log(sample_density[:, 1:, ...] + self.eps)

        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim] - 1) - scores[:, 0, ...]
        loss = L.mean()
        return loss


class KLRegressionGrid(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using the grid integration strategy."""

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0):
        """Args:
            scores: predicted score values
            gt_density: probability density of the ground truth distribution
            grid_dim: dimension(s) of the grid
            grid_scale: area of one grid cell"""

        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)

        L = torch.logsumexp(scores, dim=grid_dim) + math.log(grid_scale) - score_corr

        return L.mean()


class KLGrid(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using the grid integration strategy."""

    def forward(self, pred_density, gt_density, grid_dim=-1, grid_scale=1.0):
        """Args:
            scores: predicted score values
            gt_density: probability density of the ground truth distribution
            grid_dim: dimension(s) of the grid
            grid_scale: area of one grid cell"""
        pred_density = torch.clamp_min(pred_density, 1e-7)
        L = -(gt_density * torch.log(pred_density)).sum(dim=grid_dim)

        return L.mean()
