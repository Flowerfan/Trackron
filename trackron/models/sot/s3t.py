import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from einops import rearrange, repeat
from torchvision.ops import RoIAlign

from .build import SOT_HEAD_REGISTRY
from trackron.structures import NestedTensor
from trackron.models.box_heads import build_box_head
from trackron.config import configurable
from trackron.models.extractor import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from trackron.models.layers.position_embedding import build_position_encoding
from trackron.data.utils import normalize_boxes


@SOT_HEAD_REGISTRY.register()
class S3THead(nn.Module):

  @configurable
  def __init__(
      self,
      *,
      feature_layer: str,
      input_proj: nn.Module,
      pos_emb: nn.Module,
      target_s_decoder: nn.Module,
      target_t_encoder: nn.Module,
      search_st_encoder: nn.Module,
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
    self.target_s_decoder = target_s_decoder
    self.target_t_encoder = target_t_encoder
    self.search_st_encoder = search_st_encoder
    self.roi_align = roi_align
    self.query_embed = query_emb

  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    hidden_dim = cfg.FEATURE_DIM
    pos_emb = build_position_encoding(cfg.POSITION_EMBEDDING, hidden_dim)
    in_cha = in_feature_channels[cfg.FEATURE_LAYER]
    input_proj = nn.Conv2d(in_cha, hidden_dim, kernel_size=1)
    st_enc_layer = TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=cfg.SPATIAL_TEMPORAL_ENCODER.HEADS,
        dim_feedforward=cfg.SPATIAL_TEMPORAL_ENCODER.DIM_FEEDFORWARD,
        dropout=cfg.SPATIAL_TEMPORAL_ENCODER.DROPOUT,
        activation=cfg.SPATIAL_TEMPORAL_ENCODER.NORM,
        normalize_before=cfg.SPATIAL_TEMPORAL_ENCODER.PRE_NORM)
    search_st_encoder = TransformerEncoder(
        encoder_layer=st_enc_layer,
        num_layers=cfg.SPATIAL_TEMPORAL_ENCODER.NUM_LAYERS,
        norm=None)

    target_s_dec_layer = TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=cfg.TARGET_SPATIAL_DECODER.HEADS,
        dim_feedforward=cfg.TARGET_SPATIAL_DECODER.DIM_FEEDFORWARD,
        dropout=cfg.TARGET_SPATIAL_DECODER.DROPOUT,
        activation=cfg.TARGET_SPATIAL_DECODER.NORM,
        normalize_before=cfg.TARGET_SPATIAL_DECODER.PRE_NORM)
    target_s_decoder = TransformerDecoder(
        decoder_layer=target_s_dec_layer,
        num_layers=cfg.TARGET_SPATIAL_DECODER.NUM_LAYERS,
        norm=nn.LayerNorm(hidden_dim))

    target_t_enc_layer = TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=cfg.TARGET_TEMPORAL_ENCODER.HEADS,
        dim_feedforward=cfg.TARGET_TEMPORAL_ENCODER.DIM_FEEDFORWARD,
        dropout=cfg.TARGET_TEMPORAL_ENCODER.DROPOUT,
        activation=cfg.TARGET_TEMPORAL_ENCODER.NORM,
        normalize_before=cfg.TARGET_TEMPORAL_ENCODER.PRE_NORM)
    target_t_encoder = TransformerEncoder(
        encoder_layer=target_t_enc_layer,
        num_layers=cfg.TARGET_TEMPORAL_ENCODER.NUM_LAYERS,
        norm=None)

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
        "target_s_decoder": target_s_decoder,
        "target_t_encoder": target_t_encoder,
        "search_st_encoder": search_st_encoder,
        "box_head": build_box_head(cfg.BOX_HEAD),
        "refine_head": refine_head,
        "roi_align": roi_align,
        "query_emb": query_emb,
    }

  def forward(self, features, boxes, masks=None):
    """predict target boxes in features given the boxes in the first frame

    Args:
        features ([BT, C, H, W]): [batch and temporal dimension are flattened]
        boxes ([B T 4]): [groudtruth target boxes in features] xyxy_abs format
        masks ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    B, T = boxes.shape[:2]
    features = self.input_proj(features[self.feature_layer])
    masks, posembs = self.get_feature_maskposemb(features, masks.flatten(0, 1))
    ### search feature spatial temporal encoding
    features = rearrange(features, '(b t) c h w -> b t c h w', b=B)
    masks = rearrange(masks, '(b t) h w -> b t h w', b=B)
    posembs = rearrange(posembs, '(b t) c h w -> b t c h w', b=B)
    search_feat = self.spatial_temporal_encode(features,
                                               search_mask=masks,
                                               search_posemb=posembs) ## B T C H W

    init_target_feat = self.extract_target_feat(search_feat[:, 0], boxes[:, 0]) ### B C
    target_feat = self.spatial_target_decode(init_target_feat,
                                             search_feat,
                                             search_posemb=posembs,
                                             search_mask=masks)

    # Run the box decoder module
    search_feat = search_feat.flatten(0,1)
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
    abs_proposals = normalize_boxes(proposals, image_size, out_format='xyxy', reverse=True).permute(1, 0, 2)
    # target_feat = rearrange(target_feat, 'N B C -> B N C')
    if self.refine_head is not None:
      results = self.refine_head(search_feat, target_feat.permute(1,0,2), abs_proposals)
    else:
      results = {'boxes': abs_proposals}
    if output_norm_box:
      pred_boxes = [normalize_boxes(boxes.permute(1,0,2), image_size, out_format='xyxy') for boxes in results['boxes']]
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

  def spatial_temporal_encode(self,
                              search_feat,
                              search_mask=None,
                              search_posemb=None):
    B, T, C, H, W = search_feat.shape
    search_feat = rearrange(search_feat, 'b t c h w -> (t h w) b c')
    if search_mask is not None:
      search_mask = rearrange(search_mask, 'b t h w -> b (t h w)')
    if search_posemb is not None:
      search_posemb = rearrange(search_posemb, 'b t c h w -> (t h w) b c')
    search_feat = self.search_st_encoder(search_feat,
                                         src_key_padding_mask=search_mask,
                                         pos=search_posemb)
    search_feat = rearrange(search_feat,
                            '(t h w) b c -> b t c h w',
                            t=T,
                            h=H,
                            w=W)
    return search_feat
  
  def spatial_target_decode(self, target_feat, search_feat, search_mask=None, search_posemb=None):
    """[summary]

    Args:
        target_feat ([B C]): target feature
        search_feat ([B T C H W]): [description]
        search_mask ([B T H W], optional): [description]. Defaults to None.
        search_posemb ([B T C H W], optional): [description]. Defaults to None.

    Returns:
        [type]: [target feature for each tracking frame]
    """
    B, T, C, H, W = search_feat.shape
    ### prepare query
    target_feat = repeat(target_feat, 'b c -> (b t) c', t=T)[None]
    query_pos = repeat(self.query_embed.weight, 'n c -> n b c', b=B *
                       T) if self.query_embed is not None else None
    ### prepare key value
    search_feat = rearrange(search_feat, 'b t c h w -> (h w) (b t) c')
    search_mask = rearrange(
        search_mask, 'b t h w -> (b t) (h w)') if search_mask is not None else None
    search_posemb = rearrange(
        search_posemb, 'b t c h w -> (h w) (b t) c') if search_posemb is not None else None

    target_feat = self.target_s_decoder(
        target_feat,
        search_feat,
        memory_key_padding_mask=search_mask,
        pos=search_posemb,
        query_pos=query_pos)
    return target_feat

  def extract_target_feat(self, feat, bb):
    """[summary]

    Args:
        feat ([torch.Tensor]): [feature for extract target feature]
        bb ([torch.Tensor]): [B 4 in xyxy format]

    Returns:
        [type]: [description]
    """
    bb = [b.reshape(1, 4) for b in bb]
    target_feat = self.roi_align(feat, bb).flatten(-3)
    return target_feat

  def track(self, features, masks, ref_info=None, init_box=None):
    cur_features = self.input_proj(features[self.feature_layer])
    cur_masks, cur_poses = self.get_feature_maskposemb(cur_features, masks)
    results = {"features": cur_features, "masks": cur_masks, "posembs": cur_poses}
    assert ref_info is None or init_box is None, 'init_box should be none if ref_info is not'
    if init_box is not None:
      return results

    
    prev_features = ref_info['features']
    prev_masks = ref_info['masks']
    prev_poses = ref_info['posembs']
    prev_boxes = ref_info['boxes']
    feats = torch.cat([prev_features, cur_features], dim=0)
    masks = torch.cat([prev_masks, cur_masks], dim=0)
    posembs = torch.cat([prev_poses, cur_poses], dim=0)

    feats = self.spatial_temporal_encode(feats[None], masks[None],
                                         posembs[None]).squeeze()

    target_feat = self.get_target_feat(feats,
                                       prev_boxes,
                                       masks=masks,
                                       posembs=posembs)
    pred_boxes = self.box_head(target_feat, feats[-1:])
    pred_boxes = self.refine_boxes(feats[-1:], target_feat, pred_boxes, output_norm_box=False)[-1]
      
    results['pred_boxes'] = pred_boxes.reshape(-1, 4).mean(0)
    return results

  def get_target_feat(self, feats, boxes, masks=None, posembs=None):
    """[summary]

    Args:
        feats ([N C H W]): [N frame features]
        masks ([N H W]): [N frame masks]
        posembs ([N C H W]): [N frame position embeddings]
        boxes ([num_target 4]): [target boxes in first num target frames, num_target < N] xyxy format

    Returns:
        target_feat [num_target, 1, C]: target features
    """
    num_target = len(boxes)
    ### query
    target_feat = self.extract_target_feat(feats[:num_target], boxes[:num_target]).unsqueeze(1)
    query_pos = repeat(self.query_embed.weight, 'b c -> n b c',
                        n=num_target) if self.query_embed is not None else None
    ### search k,v
    search_feat = rearrange(feats[-1:], 'b c h w -> (h w) b c')
    search_posemb = rearrange(posembs[-1:], 'b c h w -> (h w) b c')
    search_mask = masks[-1:].flatten(-2)
    target_feat = self.target_s_decoder(
        target_feat,
        search_feat,
        memory_key_padding_mask=search_mask,
        pos=search_posemb,
        query_pos=query_pos)
    return target_feat


@SOT_HEAD_REGISTRY.register()
class SiameseS3THead(S3THead):

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
    ### search feature spatial temporal encoding
    features = torch.stack([template_features, search_features], dim=1)
    masks = torch.stack([template_masks, search_masks], dim=1)
    posembs = torch.stack([template_posembs, search_posembs], dim=1)
    boxes = torch.stack([template_boxes, search_boxes], dim=1)
    search_feat = self.spatial_temporal_encode(features,
                                               search_mask=masks,
                                               search_posemb=posembs)

    init_target_feat = self.extract_target_feat(search_feat[:, 0], boxes[:, 0])
    target_feat = self.spatial_target_decode(init_target_feat,
                                             search_feat,
                                             search_posemb=posembs,
                                             search_mask=masks)

    # Run the box decoder module
    # search_feat = search_feat[:, 1:, ...]
    # target_feat = rearrange(target_feat, 'k (b t) c -> b k t c', b=len(search_feat))[1]
    search_feat = rearrange(search_feat, 'b t c h w -> (h w) (b t) c')
    pred_boxes = self.box_head(target_feat, search_feat)
    if self.refine_head is not None:
      pred_boxes = self.refine_boxes(search_feat, target_feat, pred_boxes)
    return {"pred_boxes": pred_boxes}


@SOT_HEAD_REGISTRY.register()
class SiameseS3TMultiQuery(SiameseS3THead):
  """multiple query for target 

  """

  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    hidden_dim = cfg.FEATURE_DIM
    pos_emb = build_position_encoding(cfg.POSITION_EMBEDDING, hidden_dim)
    in_cha = in_feature_channels[cfg.FEATURE_LAYER]
    input_proj = nn.Conv2d(in_cha, hidden_dim, kernel_size=1)
    st_enc_layer = TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=cfg.SPATIAL_TEMPORAL_ENCODER.HEADS,
        dim_feedforward=cfg.SPATIAL_TEMPORAL_ENCODER.DIM_FEEDFORWARD,
        dropout=cfg.SPATIAL_TEMPORAL_ENCODER.DROPOUT,
        activation=cfg.SPATIAL_TEMPORAL_ENCODER.NORM,
        normalize_before=cfg.SPATIAL_TEMPORAL_ENCODER.PRE_NORM)
    search_st_encoder = TransformerEncoder(
        encoder_layer=st_enc_layer,
        num_layers=cfg.SPATIAL_TEMPORAL_ENCODER.NUM_LAYERS,
        norm=None)

    target_s_dec_layer = TransformerDecoderLayer(
        d_model=hidden_dim,
        nhead=cfg.TARGET_SPATIAL_DECODER.HEADS,
        dim_feedforward=cfg.TARGET_SPATIAL_DECODER.DIM_FEEDFORWARD,
        dropout=cfg.TARGET_SPATIAL_DECODER.DROPOUT,
        activation=cfg.TARGET_SPATIAL_DECODER.NORM,
        normalize_before=cfg.TARGET_SPATIAL_DECODER.PRE_NORM)
    target_s_decoder = TransformerDecoder(
        decoder_layer=target_s_dec_layer,
        num_layers=cfg.TARGET_SPATIAL_DECODER.NUM_LAYERS,
        norm=nn.LayerNorm(hidden_dim))

    target_t_enc_layer = TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=cfg.TARGET_TEMPORAL_ENCODER.HEADS,
        dim_feedforward=cfg.TARGET_TEMPORAL_ENCODER.DIM_FEEDFORWARD,
        dropout=cfg.TARGET_TEMPORAL_ENCODER.DROPOUT,
        activation=cfg.TARGET_TEMPORAL_ENCODER.NORM,
        normalize_before=cfg.TARGET_TEMPORAL_ENCODER.PRE_NORM)
    target_t_encoder = TransformerEncoder(
        encoder_layer=target_t_enc_layer,
        num_layers=cfg.TARGET_TEMPORAL_ENCODER.NUM_LAYERS,
        norm=None)

    query_emb = nn.Embedding(1, hidden_dim) if cfg.USE_QUERY_EMB else None
    kernel_sz = cfg.KERNEL_SZ
    stride = cfg.FEATURE_STRIDE
    roi_align = nn.ModuleList([
        RoIAlign(1, 1 / stride, 2, aligned=True),
        RoIAlign(3, 1 / stride, 2, aligned=True)
    ])

    return {
        "pos_emb": pos_emb,
        "feature_layer": cfg.FEATURE_LAYER,
        "input_proj": input_proj,
        "target_s_decoder": target_s_decoder,
        "target_t_encoder": target_t_encoder,
        "search_st_encoder": search_st_encoder,
        "box_head": build_box_head(cfg.BOX_HEAD),
        "roi_align": roi_align,
        "query_emb": query_emb,
        "target_token": cfg.TARGET_TOKEN
    }

  def forward(self,
              template_features,
              template_boxes,
              search_features,
              search_boxes,
              template_masks=None,
              search_masks=None):
    template_features = self.input_proj(template_features[self.feature_layer])
    template_masks, template_posembs = self.get_feature_maskposemb(
        template_features, template_masks)
    search_features = self.input_proj(search_features[self.feature_layer])
    search_masks, search_posembs = self.get_feature_maskposemb(
        search_features, search_masks)
    ### search feature spatial temporal encoding
    features = torch.stack([template_features, search_features], dim=1)
    masks = torch.stack([template_masks, search_masks], dim=1)
    posembs = torch.stack([template_posembs, search_posembs], dim=1)
    boxes = torch.stack([template_boxes, search_boxes], dim=1)
    search_feat = self.spatial_temporal_encode(features,
                                               search_mask=masks,
                                               search_posemb=posembs)

    target_feat = self.spatial_target_decode(search_feat,
                                             boxes,
                                             pos=posembs,
                                             mask=masks)

    # Run the box decoder module
    pred_boxes = self.box_head(target_feat, search_feat.flatten(0,1))
    return {"pred_boxes": pred_boxes.mean(0)}

  def spatial_target_decode(self, feat, bb, mask=None, pos=None):
    """[summary]

    Args:
        feat ([B T C H W]): [description]
        bb ([B T 4]): [description]
        mask ([B T H W], optional): [description]. Defaults to None.
        pos ([B T C H W], optional): [description]. Defaults to None.

    Returns:
        [type]: [target feature for each tracking frame]
    """
    B, T, C, H, W = feat.shape
    query_pos = repeat(self.query_embed.weight, 'n c -> n b c', b=B *
                       T) if self.query_embed is not None else None
    memory = rearrange(feat, 'b t c h w -> (h w) (b t) c')
    memory_mask = rearrange(
        mask, 'b t h w -> (b t) (h w)') if mask is not None else None
    memory_pos = rearrange(
        pos, 'b t c h w -> (h w) (b t) c') if pos is not None else None
    temp_feat = feat[:, 0]
    temp_box = bb[:, 0]
    temp_mask = mask[:, 0] if mask is not None else None
    temp_posemb = pos[:, 0] if pos is not None else None
    target_feat, target_pos = self.extract_target_feat(temp_feat, temp_box,
                                                       temp_posemb)
    tgt = repeat(target_feat, 'b c n -> n (b t) c', t=T)
    # decode_target_feat = init_target_feat[None]
    decode_target_feat = self.target_s_decoder(
        tgt,
        memory,
        memory_key_padding_mask=memory_mask,
        pos=memory_pos,
        query_pos=query_pos)
    return decode_target_feat

  def extract_target_feat(self, feat, bb, pos_emb=None):
    bb = [b.reshape(1, 4) for b in bb]
    target_feats = []
    target_poses = [] if pos_emb is not None else None
    for roi_align in self.roi_align:
      target_feats += [roi_align(feat, bb).flatten(-2)]
      if pos_emb is not None:
        target_poses += [roi_align(pos_emb, bb).flatten(-2)]
    target_feats = torch.cat(target_feats, -1)
    if pos_emb is not None:
      target_poses = torch.cat(target_poses, -1)

    return target_feats, target_poses

  def get_target_feat(self, feats, boxes, num_train, masks=None, posembs=None):
    """[summary]

    Args:
        feats ([type]): [description]
        masks ([type]): [description]
        posembs ([type]): [description]
        boxes ([type]): [description]
        num_train ([type]): [description]

    Returns:
        [type]: [description]
    """

    temp_feat = feats[:num_train]
    temp_box = boxes[:num_train]
    temp_mask = masks[:num_train] if masks is not None else None
    temp_posemb = posembs[:num_train] if posembs is not None else None
    target_feat, target_pos = self.extract_target_feat(temp_feat, temp_box,
                                                       temp_posemb)
    search_feat = rearrange(feats[-1:], 'b c h w -> (h w) b c')
    search_posemb = rearrange(posembs[-1:], 'b c h w -> (h w) b c')
    search_mask = masks[-1:].flatten(-2)
    query_pos = repeat(self.query_embed.weight, 'b c -> n b c',
                       n=num_train) if self.query_embed is not None else None
    # decode_target_feat = init_target_feat[None]
    tgt = rearrange(target_feat, 'b c n -> (b n) c').unsqueeze(1)
    decode_target_feat = self.target_s_decoder(
        tgt,
        search_feat,
        memory_key_padding_mask=search_mask,
        pos=search_posemb,
        query_pos=query_pos)
    return decode_target_feat
    
@SOT_HEAD_REGISTRY.register()
class SiameseS3TUpdateHead(S3THead):

  def forward(self,
              template_features,
              template_boxes,
              search_features,
              search_boxes,
              template_masks=None,
              search_masks=None):
    template_features = self.input_proj(template_features[self.feature_layer])
    template_masks, template_posembs = self.get_feature_maskposemb(
        template_features, template_masks)
    search_features = self.input_proj(search_features[self.feature_layer])
    search_masks, search_posembs = self.get_feature_maskposemb(
        search_features, search_masks)
    
    ### search feature spatial temporal encoding
    features = torch.stack([template_features, search_features], dim=1)
    masks = torch.stack([template_masks, search_masks], dim=1)
    posembs = torch.stack([template_posembs, search_posembs], dim=1)
    boxes = torch.stack([template_boxes, search_boxes], dim=1)
    search_feat = self.spatial_temporal_encode(features,
                                               search_mask=masks,
                                               search_posemb=posembs)

    ### extract target_feat
    init_target_feat = self.extract_target_feat(search_feat[:, 0], boxes[:, 0])
    target_feats = [init_target_feat.repeat_interleave(2, 0)[None]]

    ### update template 
    target_feat = self.spatial_target_decode(init_target_feat,
                                             search_feat,
                                             search_posemb=posembs,
                                             search_mask=masks)
    target_feats.append(target_feat)
    init_target_feat = target_feat[0, ::2]
    

    target_feat = self.spatial_target_decode(init_target_feat,
                                             search_feat,
                                             search_posemb=posembs,
                                             search_mask=masks)
    target_feats.append(target_feat)

    # Run the box decoder module
    # search_feat = search_feat[:, 1:, ...]
    # target_feat = rearrange(target_feat, 'k (b t) c -> b k t c', b=len(search_feat))[1]
    search_feat = search_feat.flatten(0, 1)

    pred_boxes = []
    for target_feat in target_feats:
      preds = self.box_head(target_feat, search_feat)
      pred_boxes.append(preds)
      if self.refine_head is not None:
        pred_boxes.extend(self.refine_boxes(search_feat, target_feat, preds)[1:])
    return {"pred_boxes": pred_boxes}

  def get_target_feat(self, feats, boxes, masks=None, posembs=None):
    """[summary]

    Args:
        feats ([type]): [description]
        masks ([type]): [description]
        posembs ([type]): [description]
        boxes ([type]): [description]

    Returns:
        [type]: [description]
    """
    num_target = len(boxes)
    ### search k,v
    search_feat = rearrange(feats, 'b c h w -> (h w) b c')
    search_posemb = rearrange(posembs, 'b c h w -> (h w) b c')
    search_mask = masks.flatten(-2)
    query_pos = repeat(self.query_embed.weight, 'b c -> n b c',
                        n=num_target) if self.query_embed is not None else None

    ### refine self target
    target_feat = self.extract_target_feat(feats[:num_target], boxes[:num_target]).unsqueeze(0)
    target_posemb = query_pos[None, :num_target, 0] if query_pos is not None else None
    target_feat = self.target_s_decoder(
        target_feat,
        search_feat[:, :num_target],
        memory_key_padding_mask=search_mask[:num_target],
        pos=search_posemb[:, :num_target],
        query_pos=target_posemb)

    ### target_feat in curret frame
    target_feat = self.target_s_decoder(
        rearrange(target_feat, 'n b c -> b n c'),
        search_feat[:, -1:],
        memory_key_padding_mask=search_mask[-1:],
        pos=search_posemb[:, -1:],
        query_pos=query_pos)
    return target_feat