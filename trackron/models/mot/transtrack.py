import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


from trackron.config import configurable
from trackron.models.extractor import DeformableTransformer
from trackron.models.layers.position_embedding import build_position_encoding
from trackron.utils.misc import inverse_sigmoid, clone_modules
from trackron.structures import NestedTensor
from trackron.models.box_heads import build_box_head

from .build import MOT_HEAD_REGISTRY





@MOT_HEAD_REGISTRY.register()
class TransTrack(nn.Module):

  @configurable
  def __init__(self,
               *,
               feature_layers: list,
               pos_emb: nn.Module,
               transformer: nn.Module,
               input_proj: nn.Module,
               combine: nn.Module,
               bbox_embed: nn.Module,
               class_embed: nn.Module,
               num_feature_levels: int,
               query_emb: Optional[nn.Module] = None,
               target_token: bool = False,
               two_stage: bool = False):
    """[summary]

    Args:
        classification_layers (list): [description]
        output_layers (list): [description]
        backbone (nn.Module): [description]
        cls_head (nn.Module): [description]
        bbox_embed (nn.Module): [description]
        pixel_mean (Tuple[float]): [description]
        pixel_std (Tuple[float]): [description]
    """
    super().__init__()
    self.feature_layers = feature_layers
    self.pos_emb = pos_emb
    self.input_proj = input_proj
    self.combine = combine
    self.query_embed = query_emb
    self.target_token = target_token
    self.transformer = transformer
    self.two_stage = two_stage
    self.bbox_embed = bbox_embed
    self.class_embed = class_embed
    self.num_feature_levels = num_feature_levels

  @classmethod
  def from_config(cls, cfg, in_feature_channels):
    feature_layers = cfg.FEATURE_LAYERS
    hidden_dim = cfg.FEATURE_DIM
    num_classes = cfg.NUM_CLASS
    pos_emb = build_position_encoding(cfg.POSITION_EMBEDDING, hidden_dim)
    out_channels = [
        in_feature_channels[layer] for layer in feature_layers
    ]

    input_proj = [
        nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        ) for in_channels in out_channels
    ]
    input_proj += [
        nn.Sequential(
            nn.Conv2d(out_channels[-1],
                      hidden_dim,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.GroupNorm(32, hidden_dim),
        )
    ]
    input_proj = nn.ModuleList(input_proj)
    combine = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

    ### For MOT
    num_feature_levels = cfg.TRANSFORMER.NUM_FEATURE_LEVELS
    transformer = DeformableTransformer(
        d_model=hidden_dim,
        nhead=cfg.TRANSFORMER.HEADS,
        num_encoder_layers=cfg.TRANSFORMER.ENC_LAYERS,
        num_decoder_layers=cfg.TRANSFORMER.DEC_LAYERS,
        dim_feedforward=cfg.TRANSFORMER.DIM_FEEDFORWARD,
        dropout=cfg.TRANSFORMER.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=cfg.TRANSFORMER.DEC_POINTS,
        enc_n_points=cfg.TRANSFORMER.ENC_POINTS,
        two_stage=cfg.TWO_STAGE,
        two_stage_num_proposals=cfg.TRANSFORMER.NUM_QUERIES,
        checkpoint_enc_ffn=False,
        checkpoint_dec_ffn=False)

    bbox_embed = build_box_head(cfg.BOX_HEAD)
    class_embed = nn.Linear(hidden_dim, cfg.NUM_CLASS)
    class_embed.bias.data = -torch.ones(num_classes) * 4.6
    two_stage = cfg.TWO_STAGE
    num_pred = cfg.TRANSFORMER.DEC_LAYERS + 1 if two_stage else cfg.TRANSFORMER.DEC_LAYERS
    num_queries = cfg.TRANSFORMER.NUM_QUERIES
    if cfg.BOX_REFINE:
      class_embed = clone_modules(class_embed, num_pred)
      bbox_embed = clone_modules(bbox_embed, num_pred)
      transformer.decoder.bbox_embed = bbox_embed
    else:
      class_embed = nn.ModuleList([class_embed for _ in range(num_pred)])
      bbox_embed = nn.ModuleList([bbox_embed for _ in range(num_pred)])
    query_embeds = nn.Embedding(num_queries, hidden_dim * 2)
    if two_stage:
      query_embeds = None
      transformer.decoder.class_embed = class_embed

    return {
        "query_emb": query_embeds,
        "feature_layers": feature_layers,
        "pos_emb": pos_emb,
        "input_proj": input_proj,
        "combine": combine,
        "transformer": transformer,
        "bbox_embed": bbox_embed,
        "class_embed": class_embed,
        "two_stage": two_stage,
        "num_feature_levels": num_feature_levels,
    }

  def forward(self, temp_feat, search_feat, temp_mask, search_mask):
    # if not self.training:
    # def forward(self, template_imgs, search_imgs, template_bb, search_proposals, *args, **kwargs):
    """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            template_imgs:  Train image samples (images, sequences, 3, H, W).
            search_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            search_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            :  Classification scores on the test samples.  """

    # Extract backbone features
    temp_srcs, temp_masks, temp_poses = self.get_mot_features(
        temp_feat, temp_feat, temp_mask)
    search_srcs, search_masks, search_poses = self.get_mot_features(
        search_feat, temp_feat, search_mask)
    # del images

    ### detection
    query_embeds = None
    if not self.two_stage:
      query_embeds = self.query_embed.weight
    #### template detection
    temp_target_feat, init_reference, inter_references, temp_class, temp_coord, _ = self.transformer(
        temp_srcs, temp_masks, temp_poses, query_embeds)
    temp_logits, temp_boxes = self.decode_mot_cls_box(temp_target_feat,
                                                      init_reference,
                                                      inter_references)
    out = {'pred_logits': temp_logits, 'pred_boxes': temp_boxes}

    #### search detection
    search_target_feat, init_reference, inter_references, search_class, search_coord, _ = self.transformer(
        search_srcs, search_masks, search_poses, query_embeds, temp_boxes[-1],
        temp_target_feat[-1])
    out['track_logits'], out['track_boxes'] = self.decode_mot_cls_box(
        search_target_feat, init_reference, inter_references)
    return out



  def get_feature_maskposemb(self, feat, mask):
    C, H, W = feat.shape[1:]
    mask = F.interpolate(mask.float(), size=(H, W)).to(torch.bool).flatten(0, 1)
    nested_feat = NestedTensor(feat, mask)
    pos_emb = self.pos_emb(nested_feat).to(feat.dtype)
    return mask, pos_emb

  def get_mot_features(self, feat1, feat2, feat_mask):
    srcs = []
    masks = []
    poses = []
    for idx, layer in enumerate(feat1.keys()):
      layer_feat1 = self.input_proj[idx](feat1[layer])
      layer_feat2 = self.input_proj[idx](feat2[layer])
      feat = self.combine(torch.cat([layer_feat1, layer_feat2], dim=1))
      mask, pos = self.get_feature_maskposemb(feat, feat_mask[None])
      srcs += [feat]
      masks += [mask]
      poses += [pos]

    srcs += [
        self.combine(
            torch.cat([self.input_proj[-1](feat1[layer]), self.input_proj[-1](feat2[layer])],
                      dim=1))
    ]
    mask, pos = self.get_feature_maskposemb(srcs[-1], feat_mask[None])
    masks += [mask]
    poses += [pos]
    return srcs, masks, poses


  def decode_mot_cls_box(self, target_feat, init_reference, inter_references):
    outputs_classes = []
    outputs_coords = []
    for lvl in range(target_feat.shape[0]):
      if lvl == 0:
        reference = init_reference
      else:
        reference = inter_references[lvl - 1]
      reference = inverse_sigmoid(reference)
      outputs_class = self.class_embed[lvl](target_feat[lvl])
      tmp = self.bbox_embed[lvl](target_feat[lvl])
      if reference.shape[-1] == 4:
        tmp += reference
      else:
        assert reference.shape[-1] == 2
        tmp[..., :2] += reference
      outputs_coord = tmp.sigmoid()
      outputs_classes.append(outputs_class)
      outputs_coords.append(outputs_coord)
    outputs_class = torch.stack(outputs_classes)
    outputs_coord = torch.stack(outputs_coords)
    return outputs_class, outputs_coord

  @torch.jit.unused
  def _set_aux_loss(self, outputs_class, outputs_coord):
    # this is a workaround to make torchscript happy, as torchscript
    # doesn't support dictionary with non-homogeneous values, such
    # as a dict having both a Tensor and a list.
    return [{
        'pred_logits': a,
        'pred_boxes': b
    } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

  @property
  def device(self):
    return self.pixel_mean.device


  def track(self, feature, mask, ref_info=None):
    pre_feat = ref_info.get('feat', feature)
    srcs, masks, pos = self.get_mot_features(feature, pre_feat, mask)

    # detection mode
    query_embeds = None
    if not self.two_stage:
      query_embeds = self.query_embed.weight
    hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory = self.transformer(
        srcs, masks, pos, query_embeds)
    cur_hs = hs
    outputs_class, outputs_coord = self.decode_mot_cls_box(
        hs, init_reference, inter_references)

    cur_class = outputs_class[-1]
    cur_box = outputs_coord[-1]
    cur_reference = cur_box
    cur_tgt = cur_hs[-1]

    if ref_info.get('reference', None) is not None:
      # track mode
      pre_reference, pre_tgt = ref_info['reference'], ref_info['tgt']

      hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, _ = self.transformer(
          srcs, masks, pos, query_embeds, pre_reference, pre_tgt, memory)
      outputs_class, outputs_coord = self.decode_mot_cls_box(
          hs, init_reference, inter_references)

      pre_class, pre_box = outputs_class[-1], outputs_coord[-1]

    else:
      pre_class, pre_box = cur_class, cur_box

    out = {
        'pred_logits': cur_class,
        'pred_boxes': cur_box,
        'detection_embs': cur_tgt,
        'tracking_logits': pre_class,
        'tracking_boxes': pre_box
    }

    ref_info = {'reference': cur_reference, 'tgt': cur_tgt, 'feat': feature}

    if self.two_stage:
      enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
      out['enc_outputs'] = {
          'pred_logits': enc_outputs_class,
          'pred_boxes': enc_outputs_coord
      }
    return out, ref_info
