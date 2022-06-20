import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath
from trackron.config import configurable
from torchvision.ops import box_convert

from .build import BOX_HEAD_REGISTRY


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  dilation=dilation,
                  bias=True), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


def mlp(in_planes, out_planes):
    return nn.Sequential(nn.Linear(in_planes, out_planes),
                         nn.LayerNorm(out_planes), nn.ReLU(inplace=True))


@BOX_HEAD_REGISTRY.register()
class Center(nn.Module):

    @configurable
    def __init__(self, input_dim, hidden_dim, dynamic_dim=64, patch_dim=484):
        super().__init__()
        '''center cord'''
        self.conv1 = conv(input_dim, hidden_dim)
        self.conv2 = conv(hidden_dim, hidden_dim // 2)
        self.conv3 = conv(hidden_dim // 2, hidden_dim // 4)
        self.conv4 = conv(hidden_dim // 4, hidden_dim // 8)
        self.conv5 = nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)

        self.layer_sz = Size(hidden_dim, dynamic_dim, patch_dim)

    @classmethod
    def from_config(cls, cfg):
        return {
            "input_dim": cfg.INPUT_DIM,
            "hidden_dim": cfg.HIDDEN_DIM,
            "dynamic_dim": cfg.DYNAMIC_DIM,
            "patch_dim": cfg.PATCH_DIM
        }

    def forward(self, target_feat, search_feat, return_score=False):
        """[summary]

    Args:
        target_feat ([N B C]): [target feature]
        search_feat ([K B C]): [search feature, must be square]
        return_score (bool, optional): [Return intermediate socres]. Defaults to False.

    Returns:
        [type]: [description]
    """
        sz = int(math.sqrt(search_feat.shape[0]))
        correlation = torch.einsum('nbc,kbc->kbn', target_feat, search_feat)
        center_feat = (correlation.unsqueeze(-1) *
                       search_feat.unsqueeze(-2)).contiguous()
        center_feat = rearrange(center_feat,
                                '(h w) b n c -> (b n) c h w',
                                h=sz,
                                w=sz)
        score_map, center = self.predict_center(center_feat)
        size = self.layer_sz(target_feat, search_feat)
        pred_boxes = torch.cat([center, size], dim=-1)
        pred_boxes = rearrange(pred_boxes, '(b n) c -> n b c', n=len(target_feat))
        pred_boxes = box_convert(pred_boxes, 'cxcywh', 'xyxy')
        if return_score:
            return score_map, pred_boxes
        return pred_boxes

    def predict_center(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        score_map = self.conv5(x).squeeze(1)
        center = self.process_center(score_map)
        return score_map, center

    def process_center(self, score):
        """[summary]

    Args:
        score ([B H W]): [box center score map]

    Returns:
        [box_]: [description]
    """
        h, w = score.shape[-2:]
        with torch.no_grad():
            # generate mesh-grid
            # indice = torch.arange(0.5, sz, dtype=score.dtype, device=score.device).view(-1, 1) / sz
            # coord_x = indice.repeat((sz, 1)).view((sz**2,))
            # coord_y = indice.repeat((1, sz)).view((sz**2,))
            coord_y, coord_x = torch.meshgrid(
                torch.linspace(0, h - 1, w, dtype=torch.float32, device=score.device),
                torch.linspace(0, w - 1, w, dtype=torch.float32, device=score.device))
            coord_x = (coord_x.flatten() + 0.5) / w
            coord_y = (coord_y.flatten() + 0.5) / h
        cx_coord, cy_coord = self.soft_argmax(score.flatten(-2), coord_x, coord_y)
        center = torch.stack([cx_coord, cy_coord], dim=1)
        return center

    def soft_argmax(self, score_map, coord_x, coord_y):
        """ get soft-argmax coordinate for a given heatmap """
        prob_vec = torch.softmax(score_map, dim=1)
        exp_x = torch.sum((coord_x * prob_vec), dim=1)
        exp_y = torch.sum((coord_y * prob_vec), dim=1)
        return exp_x, exp_y


class Size(nn.Module):

    def __init__(self, hidden_dim, dynamic_dim, patch_dim=484):
        super().__init__()
        '''center cord'''
        self.hidden_dim = hidden_dim
        self.dynamic_dim = dynamic_dim
        self.num_params = hidden_dim * dynamic_dim
        self.dynamic = nn.Linear(hidden_dim, self.num_params * 2)

        self.norm1 = nn.LayerNorm(dynamic_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU(inplace=True)


        self.layer1 = nn.Sequential(mlp(hidden_dim, hidden_dim // 4), Rearrange('b k c -> b c k'),
                                    mlp(patch_dim, patch_dim // 4), Rearrange('b c k -> b k c'))
        self.layer2 = nn.Sequential(mlp(hidden_dim // 4, hidden_dim // 16), Rearrange('b k c -> b c k'),
                                    mlp(patch_dim // 4, patch_dim // 16), Rearrange('b c k -> b k c'))
        self.layer3 = nn.Sequential(mlp(hidden_dim // 16, 8), Rearrange('b k c -> b c k'),
                                    mlp(patch_dim // 16, 8), Rearrange('b c k -> b k c'))
        self.size_layer = nn.Linear(64, 2)

    def forward(self, target_feat, search_feat):
        """[summary]

    Args:
        target_feat ([Q B C]): [target feature]
        search_feat ([K B C]): [search feature, must be square]
        return_score (bool, optional): [Return intermediate socres]. Defaults to False.

    Returns:
        [type]: [description]
    """
        q, b, c = target_feat.shape
        target_feat = rearrange(target_feat, 'q b c -> (b q) c').unsqueeze(1)
        search_feat = repeat(search_feat, 'k b c -> (b q) k c', q=q)
        params = self.dynamic(target_feat)
        param1 = params[..., :self.num_params].view(b*q, c, -1)
        param2 = params[..., self.num_params:].view(b*q, -1, c)

        features = torch.bmm(search_feat, param1)
        features = self.act(self.norm1(features))
        features = torch.bmm(features, param2)
        features = self.act(self.norm2(features))

        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        sz = self.size_layer(features.flatten(-2))
        return torch.sigmoid(sz)
