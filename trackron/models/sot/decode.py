import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from trackron.config import configurable
from trackron.models.extractor import TransformerDecoder, TransformerDecoderLayer
from trackron.models.layers.position_embedding import build_position_encoding
from einops import rearrange, repeat
from torchvision.ops import RoIAlign
from trackron.structures import NestedTensor

from .build import SOT_HEAD_REGISTRY
from trackron.models.box_heads import build_box_head
from trackron.data.utils import normalize_boxes


@SOT_HEAD_REGISTRY.register()
class DecodeHead(nn.Module):

  @configurable
  def __init__(
      self,
      *,
      feature_layer: str,
      input_proj: nn.Module,
      pos_emb: nn.Module,
      decoder: nn.Module,
      box_head: nn.Module,
      refine_head: nn.Module,
      roi_align: nn.Module,
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
    self.decoder = decoder
    self.roi_align = roi_align
    self.query_embed = query_emb

  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    hidden_dim = cfg.FEATURE_DIM
    pos_emb = build_position_encoding(cfg.POSITION_EMBEDDING, hidden_dim)
    in_cha = in_feature_channels[cfg.FEATURE_LAYER]
    input_proj = nn.Conv2d(in_cha, hidden_dim, kernel_size=1)

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
    kernel_sz = cfg.KERNEL_SZ
    stride = cfg.FEATURE_STRIDE
    roi_align = RoIAlign(kernel_sz, 1 / stride, 2, aligned=True)
    refine_head = None
    if cfg.BOX_REFINE:
      refine_head = build_box_head(cfg.REFINE_HEAD)
    return {
        "pos_emb": pos_emb,
        "feature_layer": cfg.FEATURE_LAYER,
        "input_proj": input_proj,
        "decoder": decoder,
        "box_head": build_box_head(cfg.BOX_HEAD),
        "refine_head": refine_head,
        "roi_align": roi_align,
        "query_emb": query_emb,
    }

  def forward(self, features, boxes, masks=None):
    B, T = boxes.shape[:2]
    features = self.input_proj(features[self.feature_layer])
    masks, posembs = self.get_feature_maskposemb(features, masks.flatten(0, 1))
    ### search feature spatial temporal encoding

    init_target_feat = self.extract_target_feat(features[::T], boxes[:, 0])
    init_target_feat = repeat(init_target_feat, 'b c -> (b t) c', t=T)[None]
    target_feat = self.spatial_target_decode(init_target_feat,
                                             features,
                                             search_posemb=posembs,
                                             search_mask=masks)

    # Run the box decoder module
    search_feat = features
    pred_boxes = self.box_head(target_feat, search_feat)
    if self.refine_head is not None:
      results = self.refine_boxes(search_feat, target_feat, pred_boxes)
      pred_boxes  = [pred_boxes] + results
    return {"pred_boxes": pred_boxes}

  def refine_boxes(self, search_feat, target_feat, proposals, output_norm_box=True):
    """[refine boxes]

    Args:
        search_feat ([type]): [B C H W]
        target_feat ([type]): [N B C]
        proposals ([type]): [N B 4]
    """
    B, C, H, W = search_feat.shape
    image_size = [H * 16, W * 16]
    abs_proposals = normalize_boxes(proposals, image_size, in_format='xyxy', out_format='xyxy', reverse=True).permute(1, 0, 2)
    # target_feat = rearrange(target_feat, 'N B C -> B N C')
    if self.refine_head is not None:
      results = self.refine_head(search_feat, target_feat.permute(1,0,2), abs_proposals)
    else:
      results = {'boxes': abs_proposals}
    if output_norm_box:
      pred_boxes = [normalize_boxes(boxes.permute(1,0,2), image_size, in_format='xyxy', out_format='xyxy') for boxes in results['boxes']]
    else:
      pred_boxes = results['boxes']
    return pred_boxes

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

  def spatial_target_decode(self, target_feat, search_feat, search_mask=None, search_posemb=None):
    """[summary]

    Args:
        target_feat ([1 B C]): target feature
        search_feat ([B C H W]): [description]
        search_mask ([B H W], optional): [description]. Defaults to None.
        search_posemb ([B C H W], optional): [description]. Defaults to None.

    Returns:
        [type]: [target feature for each tracking frame]
    """
    B, C, H, W = search_feat.shape
    ### prepare query
    query_pos = repeat(self.query_embed.weight, 'n c -> n b c', b=B) if self.query_embed is not None else None
    ### prepare key value
    search_feat = rearrange(search_feat, 'b c h w -> (h w) b c')
    search_mask = rearrange(
        search_mask, 'b h w -> b (h w)') if search_mask is not None else None
    search_posemb = rearrange(
        search_posemb, 'b c h w -> (h w) b c') if search_posemb is not None else None

    target_feat = self.decoder(
        target_feat,
        search_feat,
        memory_key_padding_mask=search_mask,
        pos=search_posemb,
        query_pos=query_pos)
    return target_feat

  def extract_target_feat(self, feat, bb):
    """[summary]

    Args:
        feat ([torch.Tensor]): [B C H W] feature for extract target feature
        bb ([torch.Tensor]): [B 4] in xyxy format

    Returns:
        [type]: [description]
    """
    bb = [b.reshape(1, 4) for b in bb]
    target_feat = self.roi_align(feat, bb).flatten(-3)
    return target_feat

  def track_init(self, features, boxes, masks=None):
    features = self.input_proj(features[self.feature_layer])
    init_target_feat = self.extract_target_feat(features, boxes.to(features.device))[None]
    return {"target_features": init_target_feat}

  def track(self, features, masks, ref_info=None, init_box=None):
    cur_features = self.input_proj(features[self.feature_layer])
    cur_masks, cur_poses = self.get_feature_maskposemb(cur_features, masks)
    results = {"features": cur_features, "masks": cur_masks, "posembs": cur_poses}
    assert ref_info is None or init_box is None, 'init_box should be none if ref_info is not'
    if init_box is not None:
      results['target_features'] = self.extract_target_feat(cur_features, init_box.to(cur_features.device))[None]
      return results

    target_features  = ref_info['target_features']
    target_feat = self.spatial_target_decode(target_features,
                                             cur_features,
                                             search_posemb=cur_poses,
                                             search_mask=cur_masks)

    search_feat = cur_features
    pred_boxes = self.box_head(target_feat, search_feat)
    pred_boxes = self.refine_boxes(search_feat, target_feat, pred_boxes, output_norm_box=False)[-1]
    results['pred_boxes'] = pred_boxes.reshape(-1, 4).mean(0, keepdim=True)
    return results

@SOT_HEAD_REGISTRY.register()
class SiameseDecodeHead(DecodeHead):

  def forward(self,
              template_features,
              template_boxes,
              search_features,
              search_boxes,
              template_masks=None,
              search_masks=None):
    template_features = self.input_proj(template_features[self.feature_layer])
    search_features = self.input_proj(search_features[self.feature_layer])
    template_masks, template_posembs = self.get_feature_maskposemb(
        template_features, template_masks)
    search_masks, search_posembs = self.get_feature_maskposemb(
        search_features, search_masks)
    ### search feature spatial temporal encoding
    features = torch.stack([template_features, search_features], dim=1).flatten(0,1)
    masks = torch.stack([template_masks, search_masks], dim=1).flatten(0,1)
    posembs = torch.stack([template_posembs, search_posembs], dim=1).flatten(0,1)

    init_target_feat = self.extract_target_feat(template_features, template_boxes)
    init_target_feat = repeat(init_target_feat, 'b c -> (b t) c', t=2)[None]
    target_feat = self.spatial_target_decode(init_target_feat,
                                             features,
                                             search_posemb=posembs,
                                             search_mask=masks)

    # Run the box decoder module
    # search_feat = search_feat[:, 1:, ...]
    # target_feat = rearrange(target_feat, 'k (b t) c -> b k t c', b=len(search_feat))[1]
    # search_feat = rearrange(features, 'b c h w -> (h w) b c')
    search_feat = features
    pred_boxes = self.box_head(target_feat, search_feat)
    if self.refine_head is not None:
      pred_boxes = self.refine_boxes(search_feat, target_feat, pred_boxes)
    return {"pred_boxes": pred_boxes}


@SOT_HEAD_REGISTRY.register()
class DecodeAllHead(DecodeHead):

  def forward(self, features, boxes, masks=None):
    B, T = boxes.shape[:2]
    features = self.input_proj(features[self.feature_layer])
    masks, posembs = self.get_feature_maskposemb(features, masks.flatten(0, 1))
    ### search feature spatial temporal encoding

    query_feat = repeat(features[::T], 'b c h w -> (h w) (b t) c', t=T)
    query_feat = self.spatial_target_decode(query_feat,
                                             features,
                                             search_posemb=posembs,
                                             search_mask=masks)
    query_feat = rearrange(query_feat, '(h w) b c -> b c h w', w=features.shape[-1])
    target_feat = self.extract_target_feat(query_feat, repeat(boxes[:, 0], 'b c -> (b t) c', t=T))[None]

    # Run the box decoder module
    search_feat = features
    pred_boxes = self.box_head(target_feat, search_feat)
    if self.refine_head is not None:
      pred_boxes = self.refine_boxes(search_feat, target_feat, pred_boxes)
    return {"pred_boxes": pred_boxes}

  def track_init(self, features, boxes, masks=None):
    features = self.input_proj(features[self.feature_layer])
    return {"features": features, "boxes": boxes.to(features.device)}

  def track(self, features, masks, ref_info):
    cur_features = self.input_proj(features[self.feature_layer])
    cur_masks, cur_poses = self.get_feature_maskposemb(cur_features, masks)
    results = {"features": cur_features}

    query_feat  = rearrange(ref_info['features'], 'b c h w -> (h w) b c')
    boxes = ref_info["boxes"]
    query_feat = self.spatial_target_decode(query_feat,
                                             cur_features,
                                             search_posemb=cur_poses,
                                             search_mask=cur_masks)
    query_feat = rearrange(query_feat, '(h w) b c -> b c h w', w=cur_features.shape[-1])
    target_feat = self.extract_target_feat(query_feat, boxes)[None]

    search_feat = cur_features
    pred_boxes = self.box_head(target_feat, search_feat)
    pred_boxes = self.refine_boxes(search_feat, target_feat, pred_boxes, output_norm_box=False)[-1]
    results['pred_boxes'] = pred_boxes.reshape(-1, 4).mean(0, keepdim=True)
    return results