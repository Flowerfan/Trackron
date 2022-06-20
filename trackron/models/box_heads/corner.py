import math
import torch
import torch.nn as nn
from einops import rearrange
from trackron.config import configurable
from trackron.utils.misc import clone_modules
from trackron.models.layers import get_norm

from .build import BOX_HEAD_REGISTRY


def conv(in_planes,
         out_planes,
         kernel_size=3,
         stride=1,
         padding=1,
         dilation=1,
         norm='BN'):
  norm_layer = get_norm(norm, out_planes)
  return nn.Sequential(
      nn.Conv2d(in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True), norm_layer, nn.ReLU(inplace=True))


def soft_argmax(score_map, coord_x, coord_y):
  """ get soft-argmax coordinate for a given heatmap 
  score_map: [b h w]
  coord_x: [h w]
  coord_y: [h w]

  Returns:
      [type]: [description]
  """
  prob_vec = torch.softmax(score_map.flatten(-2), dim=1)
  exp_x = torch.sum((coord_x.flatten() * prob_vec), dim=1)
  exp_y = torch.sum((coord_y.flatten() * prob_vec), dim=1)
  return exp_x, exp_y


def process_box(tl_score, br_score):
  '''about coordinates and indexs'''
  height, width = tl_score.shape[-2:]
  with torch.no_grad():
    # generate mesh-grid
    coord_y, coord_x = torch.meshgrid(
        torch.linspace(
            0, height - 1, height, dtype=torch.float32, device=tl_score.device)
        / height,
        torch.linspace(
            0, width - 1, width, dtype=torch.float32, device=tl_score.device) /
        width)
  l_coord, t_coord = soft_argmax(tl_score, coord_x, coord_y)
  r_coord, b_coord = soft_argmax(br_score, coord_x, coord_y)
  box_pred = torch.stack([l_coord, t_coord, r_coord, b_coord], dim=1)
  return box_pred


@BOX_HEAD_REGISTRY.register()
class Corner(nn.Module):

  @configurable
  def __init__(self, input_dim, hidden_dim, norm='BN'):
    super().__init__()
    '''top-left corner'''
    self.conv1_tl = conv(input_dim, hidden_dim, norm=norm)
    self.conv2_tl = conv(hidden_dim, hidden_dim // 2, norm=norm)
    self.conv3_tl = conv(hidden_dim // 2, hidden_dim // 4, norm=norm)
    self.conv4_tl = conv(hidden_dim // 4, hidden_dim // 8, norm=norm)
    self.conv5_tl = nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)
    '''bottom-right corner'''
    self.conv1_br = conv(input_dim, hidden_dim, norm=norm)
    self.conv2_br = conv(hidden_dim, hidden_dim // 2, norm=norm)
    self.conv3_br = conv(hidden_dim // 2, hidden_dim // 4, norm=norm)
    self.conv4_br = conv(hidden_dim // 4, hidden_dim // 8, norm=norm)
    self.conv5_br = nn.Conv2d(hidden_dim // 8, 1, kernel_size=1)

  @classmethod
  def from_config(cls, cfg):
    return {
        "input_dim": cfg.INPUT_DIM,
        "hidden_dim": cfg.HIDDEN_DIM,
        "norm": cfg.NORM
    }

  def forward(self, target_feat, search_feat, return_score=False):
    """[summary]

    Args:
        target_feat ([N B C]): [target feature]
        search_feat ([B C H W]): [search feature, must be square]
        return_score (bool, optional): [Return intermediate socres]. Defaults to False.

    Returns:
        [type]: [description]
    """
    height, width = search_feat.shape[-2:]
    search_feat = rearrange(search_feat, 'b c h w -> (h w) b c')
    target_mask = torch.einsum('nbc,kbc->kbn', target_feat, search_feat)
    box_feat = (target_mask.unsqueeze(-1) *
                search_feat.unsqueeze(-2)).contiguous()
    box_feat = rearrange(box_feat,
                         '(h w) b n c -> (b n) c h w',
                         h=height,
                         w=width)
    tl_score, br_score = self.predict_box_score(box_feat)
    pred_boxes = process_box(tl_score, br_score)
    pred_boxes = rearrange(pred_boxes, '(b n) c -> n b c', n=len(target_feat))
    if return_score:
      return tl_score, br_score, pred_boxes
    return pred_boxes

  def predict_box_score(self, x):
    # top-left branch
    x_tl1 = self.conv1_tl(x)
    x_tl2 = self.conv2_tl(x_tl1)
    x_tl3 = self.conv3_tl(x_tl2)
    x_tl4 = self.conv4_tl(x_tl3)
    score_map_tl = self.conv5_tl(x_tl4).squeeze(1)

    # bottom-right branch
    x_br1 = self.conv1_br(x)
    x_br2 = self.conv2_br(x_br1)
    x_br3 = self.conv3_br(x_br2)
    x_br4 = self.conv4_br(x_br3)
    score_map_br = self.conv5_br(x_br4).squeeze(1)
    return score_map_tl, score_map_br


@BOX_HEAD_REGISTRY.register()
class Corner2(nn.Module):
  """output tl br map together"""

  @configurable
  def __init__(self, input_dim, hidden_dim, norm='BN'):
    super().__init__()
    self.conv1 = conv(input_dim, hidden_dim, norm=norm)
    self.conv2 = conv(hidden_dim, hidden_dim // 2, norm=norm)
    self.conv3 = conv(hidden_dim // 2, hidden_dim // 4, norm=norm)
    self.conv4 = conv(hidden_dim // 4, hidden_dim // 8, norm=norm)
    self.conv5 = nn.Conv2d(hidden_dim // 8, 2, kernel_size=1)

  @classmethod
  def from_config(cls, cfg):
    return {
        "input_dim": cfg.INPUT_DIM,
        "hidden_dim": cfg.HIDDEN_DIM,
        "norm": cfg.NORM
    }

  def forward(self, target_feat, search_feat, return_score=False):
    """[summary]

    Args:
        target_feat ([N B C]): [target feature]
        search_feat ([B C H W]): [search feature, must be square]
        return_score (bool, optional): [Return intermediate socres]. Defaults to False.

    Returns:
        [type]: [description]
    """
    height, width = search_feat.shape[-2:]
    search_feat = rearrange(search_feat, 'b c h w -> (h w) b c')
    target_mask = torch.einsum('nbc,kbc->kbn', target_feat, search_feat)
    box_feat = (target_mask.unsqueeze(-1) *
                search_feat.unsqueeze(-2)).contiguous()
    box_feat = rearrange(box_feat,
                         '(h w) b n c -> (b n) c h w',
                         h=height,
                         w=width)
    tl_score, br_score = self.predict_box_score(box_feat)
    pred_boxes = process_box(tl_score, br_score)
    pred_boxes = rearrange(pred_boxes, '(b n) c -> n b c', n=len(target_feat))
    if return_score:
      return tl_score, br_score, pred_boxes
    return pred_boxes

  def predict_box_score(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    score_map = self.conv5(x)
    return score_map[:, 0], score_map[:, 1]


@BOX_HEAD_REGISTRY.register()
class CornerLogitMean(Corner):

  def forward(self, target_feat, search_feat, return_score=False):
    height, width = search_feat.shape[-2:]
    search_feat = rearrange(search_feat, 'b c h w -> (h w) b c')
    target_mask = torch.einsum('nbc,kbc->kbn', target_feat, search_feat)
    box_feat = (target_mask.unsqueeze(-1) *
                search_feat.unsqueeze(-2)).contiguous()
    box_feat = rearrange(box_feat,
                         '(h w) b n c -> (b n) c h w',
                         h=height,
                         w=width)
    tl_score, br_score = self.predict_box_score(box_feat)
    tl_score = rearrange(tl_score.flatten(-3),
                         '(b n) k -> b n k',
                         n=len(target_feat)).mean(1)
    br_score = rearrange(br_score.flatten(-3),
                         '(b n) k -> b n k',
                         n=len(target_feat)).mean(1)
    pred_boxes = process_box(tl_score, br_score).unsqueeze(0)
    if return_score:
      return tl_score, br_score, pred_boxes
    return pred_boxes


@BOX_HEAD_REGISTRY.register()
class CornerProbMean(Corner):

  def forward(self, target_feat, search_feat, return_score=False):
    """[summary]

    Args:
        target_feat ([N B C]): [target feature]
        search_feat ([K B C]): [search feature, must be square]
        return_score (bool, optional): [Return intermediate socres]. Defaults to False.

    Returns:
        [pred_box]: [1 B 4]
    """
    sz = int(math.sqrt(search_feat.shape[0]))
    target_mask = torch.einsum('nbc,kbc->kbn', target_feat, search_feat)
    box_feat = (target_mask.unsqueeze(-1) *
                search_feat.unsqueeze(-2)).contiguous()
    box_feat = rearrange(box_feat, '(h w) b n c -> (b n) c h w', h=sz, w=sz)
    tl_score, br_score = self.predict_box_score(box_feat)
    tl_score = rearrange(tl_score.flatten(-3),
                         '(b n) k -> b n k',
                         n=len(target_feat))
    br_score = rearrange(br_score.flatten(-3),
                         '(b n) k -> b n k',
                         n=len(target_feat))
    tl_prob = torch.softmax(tl_score, dim=-1).mean(1)
    br_prob = torch.softmax(br_score, dim=-1).mean(1)
    pred_boxes = process_box(tl_prob, br_prob).unsqueeze(0)
    if return_score:
      return tl_score, br_score, pred_boxes
    return pred_boxes

  def soft_argmax(self, prob_vec, coord_x, coord_y):
    """ get soft-argmax coordinate for a given heatmap """
    exp_x = torch.sum((coord_x * prob_vec), dim=1)
    exp_y = torch.sum((coord_y * prob_vec), dim=1)
    return exp_x, exp_y


@BOX_HEAD_REGISTRY.register()
class Corner_xyxy(nn.Module):

  @configurable
  def __init__(self, layers, coord_x, coord_y):
    super().__init__()
    self.layers = layers
    self.coord_x = coord_x
    self.coord_y = coord_y

  @classmethod
  def from_config(cls, cfg):
    input_dim = cfg.INPUT_DIM
    hidden_dim = cfg.HIDDEN_DIM
    corner_layer = nn.Sequential(
        conv(input_dim, hidden_dim),
        *[conv(hidden_dim // 2**i, hidden_dim // 2**(i + 1)) for i in range(3)],
        nn.Conv2d(hidden_dim // 8, 1, kernel_size=1))
    corner_layers = clone_modules(corner_layer, 4)
    sz = int(math.sqrt(cfg.PATCH_DIM))
    with torch.no_grad():
      # generate mesh-grid
      indice = torch.arange(
          0, sz, dtype=torch.float32, device=torch.device("cuda")).view(-1,
                                                                        1) / sz
      coord_x = indice.repeat((sz, 1)).view((sz**2,))
      coord_y = indice.repeat((1, sz)).view((sz**2,))
    return {"layers": corner_layers, "coord_x": coord_x, "coord_y": coord_y}

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
    target_mask = torch.einsum('nbc,kbc->kbn', target_feat, search_feat)
    box_feat = (target_mask.unsqueeze(-1) *
                search_feat.unsqueeze(-2)).contiguous()
    box_feat = rearrange(box_feat, '(h w) b n c -> (b n) c h w', h=sz, w=sz)
    score_maps = self.predict_box_score(box_feat)
    pred_boxes = self.process_boxes(score_maps)
    pred_boxes = rearrange(pred_boxes, '(b n) c -> n b c', n=len(target_feat))
    if return_score:
      return pred_boxes, score_maps
    return pred_boxes

  def predict_box_score(self, x):
    # top-left branch
    score_maps = []
    for layer in self.layers:
      score_maps.append(layer(x))
    return score_maps

  def process_boxes(self, score_maps):
    '''about coordinates and indexs'''
    coords = []
    for score_map in score_maps:
      x_coord, y_coord = self.soft_argmax(score_map.flatten(-3))
      coords += [(x_coord, y_coord)]
    boxes = self.coords_to_boxes(coords)
    return boxes

  def coords_to_boxes(self, coords):
    x1 = coords[0][0]
    y1 = coords[1][1]
    x2 = coords[2][0]
    y2 = coords[3][1]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes

  def soft_argmax(self, score_map):
    """ get soft-argmax coordinate for a given heatmap """
    prob_vec = torch.softmax(score_map, dim=1)
    exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
    exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
    return exp_x, exp_y


@BOX_HEAD_REGISTRY.register()
class Corner_tlbr(Corner_xyxy):

  def coords_to_boxes(self, coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    x4, y4 = coords[3]
    tl_x = (x1 + x3) / 2.0
    tl_y = (y1 + y2) / 2.0
    br_x = (x2 + x4) / 2.0
    br_y = (y3 + y4) / 2.0
    boxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=-1)
    return boxes