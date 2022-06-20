import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou

from trackron.config import configurable
from trackron.data.mask_ops import paste_masks_in_image
from trackron.data.utils import generate_proposals, normalize_boxes
from trackron.models.backbone import build_backbone
from trackron.models.objectives import build_objective
from trackron.models.box_heads import build_box_head
from trackron.structures import NestedTensor, ImageList, Boxes
from trackron.models.poolers import ROIPooler
from trackron.models.layers import get_norm
from trackron.models.extractor import (TransformerDecoder,
                                       TransformerDecoderLayer,
                                       DeformableTransformerEncoder,
                                       DeformableTransformerEncoderLayer,
                                       DeformableTransformerDecoderLayer,
                                       DeformableTransformerDecoder)
from trackron.models.layers.position_embedding import build_position_encoding
from trackron.utils.misc import inverse_sigmoid

from .build import META_ARCH_REGISTRY


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def proj_layer(in_channels,
               hidden_dim,
               kernel_sz=1,
               padding=0,
               stride=1,
               norm='BN'):
    norm_layer = get_norm(norm, hidden_dim)
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  hidden_dim,
                  kernel_size=kernel_sz,
                  padding=padding,
                  stride=stride), norm_layer)
    # return nn.Sequential(nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_sz, padding=padding, stride=stride),
    #                      nn.GroupNorm(hidden_dim // 8, hidden_dim))


@META_ARCH_REGISTRY.register()
class UnifiedTransformerTracker(nn.Module):

    @configurable
    def __init__(self,
                 *,
                 backbone: nn.Module,
                 num_feature_layers: int,
                 feature_layers: list,
                 feature_strides: list,
                 pos_emb: nn.Module,
                 level_embed: nn.Module,
                 reference_points: nn.Module,
                 detection_proj: nn.Module,
                 tracking_proj: nn.Module,
                 mot_pooler: nn.Module,
                 sot_pooler: nn.Module,
                 sot_roi: nn.Module,
                 mot_roi: nn.Module,
                 target_proj: nn.Module,
                 encoder: nn.Module,
                 det_decoder: nn.Module,
                 sot_decoder: nn.Module,
                 sot_proposal: nn.Module,
                 box_head: nn.Module,
                 class_head: nn.Module,
                 track_head: nn.Module,
                 query_emb: Optional[nn.Module] = None,
                 sot_feature_layer: Optional[int] = 1,
                 mot_mask_emb: Optional[nn.Module] = None,
                 target_token: bool = False,
                 sot_objective: nn.Module,
                 mot_objective: nn.Module,
                 pixel_mean: Tuple[float],
                 pixel_std: Tuple[float]):
        """[summary]

    Args:
        classification_layers (list): [description]
        backbone (nn.Module): [description]
        cls_head (nn.Module): [description]
        box_head (nn.Module): [description]
        pixel_mean (Tuple[float]): [description]
        pixel_std (Tuple[float]): [description]
    """
        super().__init__()
        self.backbone = backbone
        self.feature_layers = feature_layers
        self.feature_strides = feature_strides
        self.pos_emb = pos_emb
        self.tracking_proj = tracking_proj

        ### encoder features
        self.level_embed = level_embed
        self.encoder = encoder
        self.query_embed = query_emb
        self.track_head = track_head
        self.box_head = box_head
        self.class_head = class_head
        self.num_feature_layers = num_feature_layers

        ### SOT
        self.sot_roi = sot_roi
        self.sot_pooler = sot_pooler
        self.sot_decoder = sot_decoder
        self.sot_proposal = sot_proposal
        self.sot_feature_layer = sot_feature_layer
        self.target_proj = target_proj

        ### MOT
        self.mot_pooler = mot_pooler
        self.mot_roi = mot_roi
        self.reference_points = reference_points
        self.detection_proj = detection_proj
        self.det_decoder = det_decoder

        ## MOTS
        self.mot_mask_emb = mot_mask_emb

        ### Objective
        self.sot_objective = sot_objective
        self.mot_objective = mot_objective

        self.register_buffer("pixel_mean",
                             torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std",
                             torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        feature_layers = cfg.MODEL.FEATURE_LAYERS
        num_feature_layers = cfg.MODEL.NUM_FEATURE_LAYERS
        hidden_dim = cfg.MODEL.FEATURE_DIM
        num_classes = cfg.MODEL.NUM_CLASS
        pos_emb = build_position_encoding(cfg.MODEL.POSITION_EMBEDDING,
                                          hidden_dim)
        backbone_out_channels = [
            backbone._out_feature_channels[layer] for layer in feature_layers
        ]

        ## Projection and RoI layers
        feature_strides = [2**(int(layer[-1]) + 1) for layer in feature_layers]
        detection_proj = [
            proj_layer(in_channel, hidden_dim, norm=cfg.MODEL.NORM)
            for in_channel in backbone_out_channels
        ]
        if num_feature_layers > len(feature_layers):
            detection_proj.append(
                proj_layer(backbone_out_channels[-1],
                           hidden_dim,
                           kernel_sz=3,
                           stride=2,
                           padding=1,
                           norm=cfg.MODEL.NORM))
            feature_strides += [feature_strides[-1] * 2]
        detection_proj = nn.ModuleList(detection_proj)
        tracking_proj = copy.deepcopy(detection_proj)

        ### For MOT
        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, cfg.MODEL.ENCODER.DIM_FEEDFORWARD,
            cfg.MODEL.ENCODER.DROPOUT, cfg.MODEL.ENCODER.NORM,
            num_feature_layers, cfg.MODEL.ENCODER.HEADS,
            cfg.MODEL.ENCODER.NUM_POINTS)
        encoder = DeformableTransformerEncoder(encoder_layer,
                                               cfg.MODEL.ENCODER.NUM_LAYERS)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, cfg.MODEL.DECODER.DIM_FEEDFORWARD,
            cfg.MODEL.DECODER.DROPOUT, cfg.MODEL.DECODER.NORM,
            num_feature_layers, cfg.MODEL.DECODER.HEADS,
            cfg.MODEL.DECODER.NUM_POINTS)
        det_decoder = DeformableTransformerDecoder(
            decoder_layer, cfg.MODEL.DECODER.NUM_LAYERS, True)
        sot_pooler = ROIPooler(
            output_size=cfg.MODEL.SOT.POOL_SIZE,
            scales=cfg.MODEL.SOT.POOL_SCALES,
            sampling_ratio=cfg.MODEL.SOT.POOL_SAMPLE_RATIO,
            pooler_type=cfg.MODEL.SOT.POOL_TYPE,
        )
        mot_pooler = ROIPooler(
            output_size=cfg.MODEL.MOT.POOL_SIZE,
            scales=cfg.MODEL.MOT.POOL_SCALES,
            sampling_ratio=cfg.MODEL.MOT.POOL_SAMPLE_RATIO,
            pooler_type=cfg.MODEL.MOT.POOL_TYPE,
        )
        osz = cfg.MODEL.OBJECT_SIZE
        sot_roi = ROIPooler(
            output_size=osz,
            scales=cfg.MODEL.SOT.POOL_SCALES,
            sampling_ratio=cfg.MODEL.SOT.POOL_SAMPLE_RATIO,
            pooler_type=cfg.MODEL.SOT.POOL_TYPE,
        )
        mot_roi = ROIPooler(
            output_size=osz,
            scales=cfg.MODEL.MOT.POOL_SCALES,
            sampling_ratio=2,
            pooler_type=cfg.MODEL.MOT.POOL_TYPE,
        )
        target_proj = nn.Sequential(nn.Linear(osz**2 * hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim))

        # sot_proposal = build_box_head(cfg.MODEL.SOT_PROPOSAL)
        #### SOT target proposal
        sot_dec_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=cfg.MODEL.SOT.DECODER.HEADS,
            dim_feedforward=cfg.MODEL.SOT.DECODER.DIM_FEEDFORWARD,
            dropout=cfg.MODEL.SOT.DECODER.DROPOUT,
            activation=cfg.MODEL.SOT.DECODER.NORM,
            normalize_before=cfg.MODEL.SOT.DECODER.PRE_NORM)
        sot_decoder = TransformerDecoder(
            decoder_layer=sot_dec_layer,
            num_layers=cfg.MODEL.SOT.DECODER.NUM_LAYERS,
            norm=nn.LayerNorm(hidden_dim))
        sot_proposal = build_box_head(cfg.MODEL.SOT.BOX_HEAD)

        ## track head
        track_head = build_box_head(cfg.MODEL.TRACK_HEAD)

        ## detection head
        box_head = build_box_head(cfg.MODEL.BOX_HEAD)
        class_head = nn.Linear(hidden_dim, cfg.MODEL.NUM_CLASS)
        class_head.bias.data = -torch.ones(num_classes) * 4.6
        num_pred = cfg.MODEL.DECODER.NUM_LAYERS
        num_queries = cfg.MODEL.NUM_QUERIES
        if cfg.MODEL.BOX_REFINE:
            class_head = _get_clones(class_head, num_pred)
            box_head = _get_clones(box_head, num_pred)
            det_decoder.bbox_embed = box_head
        else:
            class_head = nn.ModuleList([class_head for _ in range(num_pred)])
            box_head = nn.ModuleList([box_head for _ in range(num_pred)])
        query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        level_embed = nn.Parameter(torch.Tensor(num_feature_layers,
                                                hidden_dim))
        reference_points = nn.Linear(hidden_dim, 2)
        nn.init.normal_(level_embed)

        ### mots
        if cfg.MODEL.MOT.USE_SEGMENTATION:
            mot_mask_emb = build_box_head(cfg.MODEL.MOT.SEGMENTATION)
            mot_mask_emb = nn.ModuleList(
                [mot_mask_emb for _ in range(num_pred)])
        else:
            mot_mask_emb = None

        return {
            "backbone": backbone,
            "query_emb": query_embed,
            "level_embed": level_embed,
            "feature_layers": feature_layers,
            "feature_strides": feature_strides,
            "pos_emb": pos_emb,
            "detection_proj": detection_proj,
            "tracking_proj": tracking_proj,
            "mot_pooler": mot_pooler,
            "sot_pooler": sot_pooler,
            "sot_roi": sot_roi,
            "mot_roi": mot_roi,
            "target_proj": target_proj,
            "sot_decoder": sot_decoder,
            "sot_proposal": sot_proposal,
            "encoder": encoder,
            "det_decoder": det_decoder,
            "track_head": track_head,
            "reference_points": reference_points,
            "box_head": box_head,
            "mot_mask_emb": mot_mask_emb,
            "class_head": class_head,
            "num_feature_layers": num_feature_layers,
            "sot_objective": build_objective(cfg.SOT.OBJECTIVE),
            "mot_objective": build_objective(cfg.MOT.OBJECTIVE),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    def forward(self, data: List[Dict[str, torch.Tensor]], mode='sot'):
        if mode == 'sot':
            return self.forward_sot(data)
        elif mode == 'vos':
            return self.forward_sot(data)
        elif mode == 'mot':
            return self.forward_mot(data)
        else:
            raise ValueError('Not supported mode')

    def forward_sot(self, data: List[Dict[str, torch.Tensor]]):
        """for sot training
    Args:
        data (List[Dict[str, torch.Tensor]]): [description]

    """
        samples = self.preprocess_sot_inputs(data)
        batch, time = samples['search_images'].shape[:2]  ### B T C H W
        features = self.backbone(samples['search_images'].flatten(0, 1))
        srcs, masks, posembs = self.get_proj_features(
            features, samples['search_masks'].flatten(0, 1),
            self.tracking_proj)
        feature_layer = self.sot_feature_layer
        frame_feature = srcs[feature_layer]
        mask, posemb = masks[feature_layer], posembs[feature_layer]

        ### prepare tracking targets and proposals
        temp_frame_feature = frame_feature[::time]
        temp_boxes = samples['search_boxes'][:, 0]
        init_target_feat = self.get_target_roi_feat([temp_frame_feature],
                                                    temp_boxes, self.sot_roi)
        init_target_feat = repeat(init_target_feat, 'b c -> (b t) c',
                                  t=time).unsqueeze(1)
        target_feat, proposals = self.prepare_sot_target_feature_proposal(
            init_target_feat, frame_feature, mask, posemb)
        #### track
        track_results = self.tracking_target(
            srcs, proposals, target_feat,
            samples['search_image_sizes'].flatten(0, 1))

        track_boxes = [proposals] + track_results['boxes']
        out = {
            "pred_boxes": track_boxes,
            "pred_scores": track_results.get("scores", None),
            "pred_masks": track_results.get("masks", None)
        }
        return self.sot_objective(out, samples)

    def forward_mot(self, data: List[Dict[str, torch.Tensor]]):
        """[summary]
    Args:
        data (List[Dict[str, torch.Tensor]]): [description]

    """

        samples = self.preprocess_mot_inputs(data)
        targets = self.prepare_mot_targets(data)
        # Extract backbone features
        template_features = self.backbone(samples['template'].tensor)
        search_features = self.backbone(samples['search'].tensor)
        temp_srcs, temp_masks, temp_posembs = self.get_proj_features(
            template_features, samples['template'].mask, self.detection_proj)

        #### prepare features
        spatial_shapes = torch.tensor(
            [list(src.shape[-2:]) for src in temp_srcs],
            device=temp_srcs[0].device,
            dtype=torch.long)
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in temp_masks], 1)
        temp_srcs, temp_masks, temp_posembs = self.flatten_feats(
            temp_srcs, temp_masks, temp_posembs)
        temp_srcs = self.encoder(temp_srcs, spatial_shapes, valid_ratios,
                                 temp_posembs, temp_masks)

        ### detection boxes for MOT and prepare proposals for tracking
        temp_target, init_ref, inter_ref = self.decode_detection_targets(
            temp_srcs, temp_masks, spatial_shapes, valid_ratios)
        temp_detections = self.decode_detection_results(
            temp_target, init_ref, inter_ref)
        temp_logits = temp_detections['logits']
        temp_boxes = temp_detections['boxes']

        #### track
        temp_srcs, _, _ = self.get_proj_features(template_features,
                                                 samples['template'].mask,
                                                 self.tracking_proj)
        search_srcs, _, _ = self.get_proj_features(search_features,
                                                   samples['search'].mask,
                                                   self.tracking_proj)
        target_feat, proposals = self.prepare_mot_target_feature_proposal(
            temp_srcs, temp_boxes[-1], targets)
        track_results = self.tracking_target(search_srcs,
                                             proposals,
                                             target_feat,
                                             targets['image_sizes'],
                                             box_format='cxcywh')
        out = {
            'pred_logits': temp_logits,
            'pred_boxes': temp_boxes,
            'track_boxes': track_results['boxes'],
            'track_scores': track_results['scores']
        }
        return self.mot_objective(out, targets)

    def tracking_target(self,
                        search_feature,
                        proposals,
                        target_feat,
                        image_sizes,
                        box_format='xyxy',
                        box_relative=True,
                        target_mask=None,
                        tracking_mode='sot'):
        """[summary]

    Args:
        search_feature (list of [B C H W]): [tracking frame feature]
        proposals ([B N 4]): [proposal boxes in normalized coordinates]
        target_feat ([B N C]): [target features]
        image_sizes ([B 2]): [Store H W of image sizes]
        box_format (str, optional): [input boxes format, boxes are normalized]. Defaults to 'xyxy'.
        box_relative (bool, optional): [output normalized boxes if set to true]. Defaults to True.
        target_mask ([type], optional): [target mask]. Defaults to None.
        tracking_mode [str]: 'sot' or 'mot'

    Returns:
        [type]: [description]
    """
        pooler = None
        if tracking_mode == 'sot':
            search_feature = [search_feature[self.sot_feature_layer]]
            pooler = self.sot_pooler
        ### proposal is transformed to abs_xyxy format
        proposals = normalize_boxes(proposals,
                                    image_sizes,
                                    in_format=box_format,
                                    out_format='xyxy',
                                    reverse=True)
        track_results = self.track_head(search_feature,
                                        target_feat,
                                        proposals,
                                        pooler=pooler,
                                        obj_masks=target_mask)
        if box_relative:
            ### box to [0, 1] for calculating losses
            track_results['boxes'] = [
                normalize_boxes(
                    boxes,
                    image_sizes,
                    in_format='xyxy',
                    out_format=box_format,
                ) for boxes in track_results['boxes']
            ]
        return track_results

    def prepare_mot_target_feature_proposal(self, features, pred_boxes,
                                            targets):
        """Gernerate MOT proposals on the search frames based on the gt annotations
       The proposals are only for objects in both template and search frames 
    Args:
        features (list of [B C H W]))
        boxes ([B N 4] [Boxes in cxcywh format (0,1)]
        spatial_shapes ([L 2]): [Spatial shape (h, w) in each layer]
        valid_ratios ([B L 2]): [valid area in each feature map]

    Returns:
        proposals
    """
        pred_det_boxes = pred_boxes.clone().detach()
        gt_det_boxes = [target['boxes'] for target in targets['det_targets']]
        gt_search_boxes = [
            target['boxes'] for target in targets['track_targets']
        ]
        search_proposals = [
            target['proposals'] for target in targets['track_targets']
        ]
        det_track_idmap = targets['matched_indices']
        try:
            det_matched = self.iou_match(pred_det_boxes, gt_det_boxes)
        except:
            det_matched = [(range(len(det_boxes)), range(len(det_boxes)))
                           for det_boxes in gt_det_boxes]
        assert len(det_matched) == len(det_track_idmap)
        ### replace det boxes with matched track proposals
        valid_preds = []
        valid_track_boxes = []
        for idx, (pred_inds, det_inds) in enumerate(det_matched):
            valid_pred = []
            valid_boxes = []
            for pred_idx, det_idx in zip(pred_inds, det_inds):
                track_idx = det_track_idmap[idx].get(det_idx, None)
                if track_idx is not None:
                    valid_pred += [pred_idx]
                    valid_boxes += [gt_search_boxes[idx][track_idx]]
                    pred_det_boxes[idx,
                                   pred_idx] = search_proposals[idx][track_idx]
            valid_preds.append(torch.as_tensor(valid_pred, dtype=torch.int64))
            valid_track_boxes.append(torch.stack(valid_boxes))
        targets['valid_preds'] = valid_preds
        targets['valid_track_boxes'] = valid_track_boxes
        abs_boxes = normalize_boxes(pred_det_boxes,
                                    targets['image_sizes'],
                                    in_format='cxcywh',
                                    out_format='xyxy',
                                    reverse=True)
        target_feat = self.get_target_roi_feat(features, abs_boxes,
                                               self.mot_roi).view(
                                                   *abs_boxes.shape[:2], -1)
        return target_feat, pred_det_boxes

    def iou_match(self, pred_boxes, target_boxes):
        bs, num_queries = pred_boxes.shape[:2]
        out_bbox = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        sizes = [len(boxes) for boxes in target_boxes]
        tgt_bbox = torch.cat([boxes for boxes in target_boxes])
        C = -generalized_box_iou(box_convert(out_bbox, 'cxcywh', 'xyxy'),
                                 box_convert(tgt_bbox, 'cxcywh', 'xyxy'))
        C = C.view(bs, num_queries, -1).cpu()
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return indices
        # return [(torch.as_tensor(i, dtype=torch.int64),
        #          torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def prepare_mot_targets(self, data):
        det_targets = []
        track_targets = []
        matched_indices = []
        image_sizes = []
        for d in data:
            det_target = {
                'boxes':
                normalize_boxes(d['template_boxes'].to(self.device),
                                d['template_images'].shape[-2:],
                                in_format='xyxy',
                                out_format='cxcywh'),
                'labels':
                d['template_labels'].to(self.device)
            }
            track_target = {
                'boxes':
                normalize_boxes(d['search_boxes'].to(self.device),
                                d['search_images'].shape[-2:],
                                in_format='xyxy',
                                out_format='cxcywh'),
                'proposals':
                normalize_boxes(generate_proposals(d['search_boxes'].to(
                    self.device)),
                                d['search_images'].shape[-2:],
                                in_format='xyxy',
                                out_format='cxcywh').view(-1, 4),
                'labels':
                d['search_labels'].to(self.device)
            }
            if self.mot_mask_emb is not None:
                det_target['masks'] = d['template_masks'].to(self.device)
                det_target['xyxy_boxes'] = d['template_boxes'].to(self.device)
                track_target['masks'] = d['search_masks'].to(self.device)
                track_target['xyxy_boxes'] = d['search_boxes'].to(self.device)
            det_targets.append(det_target)
            track_targets.append(track_target)
            matched_indices.append(d['matched_indices'])
            image_sizes.append(d['search_images'].shape[-2:])

        targets = {
            'det_targets': det_targets,
            'track_targets': track_targets,
            'image_sizes': torch.tensor(image_sizes).float().to(self.device),
            'matched_indices': matched_indices
        }
        return targets

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def flatten_feats(self, srcs, masks, pos_embeds):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        for lvl, (src, mask,
                  pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
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
        return src_flatten, mask_flatten, lvl_pos_embed_flatten

    def unflatten_features(self, features, masks, posembs, spatial_shapes):
        hw_features = []
        hw_masks = []
        hw_posembs = []
        sidx = eidx = 0
        for shape in spatial_shapes:
            h, w = shape
            eidx = sidx + h * w
            hw_features += [
                rearrange(features[:, sidx:eidx], 'b (h w) c -> b c h w', h=h)
            ]
            hw_masks += [
                rearrange(masks[:, sidx:eidx], 'b (h w) -> b h w', h=h)
            ]
            hw_posembs += [
                rearrange(posembs[:, sidx:eidx], 'b (h w) c -> b c h w', h=h)
            ]
            sidx = eidx
        return hw_features, hw_masks, hw_posembs

    def decode_target_feature(self,
                              init_target_feat,
                              search_feat,
                              search_mask=None,
                              search_posemb=None):
        flat_search_feat = rearrange(search_feat, 'b c h w -> (h w) b c')
        flat_search_mask = rearrange(
            search_mask,
            'b h w -> b (h w)') if search_mask is not None else None
        flat_search_posemb = rearrange(
            search_posemb,
            'b c h w -> (h w) b c') if search_posemb is not None else None
        target_feat = self.sot_decoder(
            init_target_feat.permute(1, 0, 2),
            flat_search_feat,
            memory_key_padding_mask=flat_search_mask,
            pos=flat_search_posemb,
            query_pos=None)
        return target_feat

    def prepare_sot_target_feature_proposal(self,
                                            init_target_feat,
                                            search_feat,
                                            search_mask=None,
                                            search_posemb=None):
        """
        decode traget feature from search feature
        and get search proposal for the target
        """

        target_feat = self.decode_target_feature(init_target_feat, search_feat,
                                                 search_mask, search_posemb)

        proposals = self.sot_proposal(target_feat, search_feat)
        target_feat = rearrange(target_feat, 'n b c -> b n c')
        proposals = rearrange(proposals, 'n b c -> b n c')
        return target_feat, proposals

    def get_target_roi_feat(self, features, boxes, roi_func):
        # bboxes = list(torch.split(boxes, 1))
        bboxes = [Boxes(box.reshape(-1, 4)) for box in boxes.detach()]
        target_feat = roi_func(features, bboxes).flatten(-3)
        target_feat = self.target_proj(target_feat)
        return target_feat

    def get_proj_features(self, features, feature_masks, proj_func):
        srcs = []
        masks = []
        poses = []
        for idx, layer in enumerate(features.keys()):
            feat = proj_func[idx](features[layer])
            mask, pos = self.get_mask_posemb(feat, feature_masks[None])
            srcs += [feat]
            masks += [mask]
            poses += [pos]

        if len(proj_func) > len(features):
            srcs += [proj_func[-1](features[layer])]
            mask, pos = self.get_mask_posemb(srcs[-1], feature_masks[None])
            masks += [mask]
            poses += [pos]
        return srcs, masks, poses

    def detect_objects(self, features, masks):
        srcs, masks, posembs = self.get_proj_features(features, masks,
                                                      self.detection_proj)
        spatial_shapes = torch.tensor([list(src.shape[-2:]) for src in srcs],
                                      device=srcs[0].device,
                                      dtype=torch.long)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        srcs, masks, posembs = self.flatten_feats(srcs, masks, posembs)
        srcs = self.encoder(srcs, spatial_shapes, valid_ratios, posembs, masks)
        target_feat, init_ref, inter_ref = self.decode_detection_targets(
            srcs, masks, spatial_shapes, valid_ratios)
        detect_results = self.decode_detection_results(target_feat, init_ref,
                                                       inter_ref)
        return detect_results

    def decode_detection_targets(self, features, masks, spatial_shapes,
                                 valid_ratios):
        """ get detection object feature and reference points

    Args:
        features ([B K C]): [description]
        masks ([B K] ): [description]
        spatial_shapes ([L,2]): [Indicate the height and width of the feature map of each layer]
        valid_ratios ([L, 2]): [Indicate the valid area of each feature map]

    Returns:
        [type]: [description]
    """

        b, k, c = features.shape
        query_embed, tgt = torch.split(self.query_embed.weight, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(b, -1, -1)
        tgt = tgt.unsqueeze(0).expand(b, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points
        # decoder
        hs, inter_references = self.det_decoder(tgt, reference_points,
                                                features, spatial_shapes,
                                                valid_ratios, query_embed,
                                                masks)
        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out

    def decode_detection_results(self, target_feat, init_reference,
                                 inter_references):
        """target feature to detection boxes in cxcywh format

    Args:
        target_feat ([type]): [description]
        init_reference ([type]): [description]
        inter_references ([type]): [description]

    Returns:
        [type]: [description]
    """
        outputs_classes = []
        outputs_coords = []
        for lvl in range(target_feat.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_head[lvl](target_feat[lvl])
            tmp = self.box_head[lvl](target_feat[lvl])
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
        outputs = {'logits': outputs_class, 'boxes': outputs_coord}
        if self.mot_mask_emb is not None:
            outputs_vectors = []
            for lvl in range(target_feat.shape[0]):
                outputs_vector = self.mot_mask_emb[lvl](target_feat[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_mask_vector = torch.stack(outputs_vectors)
            outputs['masks'] = outputs_mask_vector
        return outputs

    def get_mask_posemb(self, feat, mask):
        C, H, W = feat.shape[1:]
        mask = F.interpolate(mask.float(),
                             size=(H, W)).to(torch.bool).flatten(0, 1)
        nested_feat = NestedTensor(feat, mask)
        pos_emb = self.pos_emb(nested_feat).to(feat.dtype)
        return mask, pos_emb

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_sot_inputs(self, batched_inputs: List[Dict[str,
                                                              torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
    """
        images = torch.stack([
            (x["search_images"].to(self.device) - self.pixel_mean[None]) /
            self.pixel_std[None] for x in batched_inputs
        ])
        image_sizes = torch.stack([
            torch.tensor(
                len(x['search_images']) *
                [[*x['search_images'].shape[-2:]]]).float().to(self.device)
            for x in batched_inputs
        ])
        boxes = torch.stack([x['search_boxes']
                             for x in batched_inputs]).to(self.device)
        att_masks = torch.stack([
            x.get(
                'search_att',
                torch.zeros(len(x['search_images']),
                            *x['search_images'].shape[-2:])).to(self.device)
            for x in batched_inputs
        ])
        target_masks = torch.stack([
            x.get(
                'search_masks',
                torch.zeros(len(x['search_images']),
                            *x['search_images'].shape[-2:])).to(self.device)
            for x in batched_inputs
        ])
        return {
            'search_images': images,
            'search_image_sizes': image_sizes,
            'search_masks': att_masks,
            'search_boxes': boxes,
            'search_target_masks': target_masks
        }

    def preprocess_mot_inputs(self, batched_inputs: List[Dict[str,
                                                              torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
    """
        samples = {}
        for s in ['template', 'search']:
            images = [(x[f"{s}_images"].to(self.device) - self.pixel_mean) /
                      self.pixel_std for x in batched_inputs]
            masks = [
                torch.zeros(*x[f'{s}_images'].shape[-2:]).to(self.device)
                for x in batched_inputs
            ]
            samples[s] = ImageList.from_tensors(images, 32, masks=masks)
        return samples

    def track_sot(self, image, mask=None, ref_info=None, init_box=None):
        image_sizes = torch.tensor([image.shape[-2:]]).float().to(self.device)
        image = (image.to(self.device) - self.pixel_mean) / self.pixel_std
        feature = self.backbone(image)
        srcs, masks, posembs = self.get_proj_features(feature, mask,
                                                      self.tracking_proj)

        if ref_info is None:
            #### for initialization
            assert init_box is not None
            target_feat = self.get_target_roi_feat([srcs[1]],
                                                   init_box.to(srcs[1].device),
                                                   self.sot_roi)[None]
            return {"target_features": target_feat, "count": 0}

        ### tracking
        target_features = ref_info['target_features']
        target_feat, proposals = self.prepare_sot_target_feature_proposal(
            target_features,
            srcs[1],
            search_posemb=posembs[1],
            search_mask=masks[1])
        # if proposals.shape[1] > 1:
        #   proposals[:, 0] = ref_info['target_boxes']
        #   proposals[:, 1] = ref_info['target_boxes']
        # proposals[:, 1] = (ref_info['target_boxes'] + proposals[:, 1]) / 2.0
        # Run the box decoder module
        tracking_results = self.tracking_target(srcs,
                                                proposals,
                                                target_feat,
                                                image_sizes,
                                                box_format='xyxy',
                                                box_relative=False)
        # pred_boxes = tracking_results['boxes'][-1].view(-1,4)
        boxes = torch.stack(tracking_results['boxes'][-3:], 0)
        results = {
            'pred_boxes':  # boxes.view(-1, 4).mean(0, keepdim=True),
            tracking_results['boxes'][-1].view(-1, 4).mean(0, keepdim=True),
            'pred_scores': 1.0
        }
        if tracking_results.get('score', None) is not None:
            results['pred_scores'] = tracking_results['scores'][-1].sigmoid(
            ).mean().item()

        if tracking_results.get('masks', None) is not None:
            results['pred_masks'] = paste_masks_in_image(
                tracking_results['masks'][-1].flatten(0, 2).sigmoid(),
                tracking_results['boxes'][-1][0], image.shape[-2:])
        return results

    @torch.no_grad()
    def track_mot(self, image, mask=None, ref_info=None):
        image = (image - self.pixel_mean[None]) / self.pixel_std[None]
        image_sizes = torch.tensor([image.shape[-2:]]).float().to(self.device)

        ### detection
        feature = self.backbone(image)
        search_feat, _, _ = self.get_proj_features(feature, mask,
                                                   self.tracking_proj)
        if ref_info.get('public_detections', None) is None:
            det_out = self.detect_objects(feature, mask)
            det_logits, det_boxes = det_out['logits'][-1], det_out['boxes'][-1]
            valid_ids = det_logits.sigmoid()[..., 1] > 0.1
            det_logits = det_logits[valid_ids][None]
            det_boxes = det_boxes[valid_ids][None]
        else:
            det_boxes = ref_info['public_detections'][..., :4]
            det_logits = ref_info['public_detections'][..., 4]

        if ref_info.get('proposal', None) is not None:
            # track mode
            pre_proposals, pre_target_feat = ref_info['proposal'], ref_info[
                'target_feat']
            results = self.tracking_target(search_feat,
                                           pre_proposals,
                                           pre_target_feat,
                                           image_sizes,
                                           box_format='cxcywh')
            track_boxes = results['boxes'][-1]
            track_scores = results.get('scores', [None])[-1]
            # track_boxes = pre_proposals
            # track_scores = None
        else:
            track_boxes = det_boxes
            track_scores = None

        out = {
            'pred_logits': det_logits,
            'pred_boxes': det_boxes,
            'tracking_logits': det_logits,
            'tracking_boxes': track_boxes,
            'tracking_scores': track_scores,
        }
        target_feat = torch.zeros_like(det_boxes)

        ### for next frame tracking
        boxes_abs = normalize_boxes(det_boxes,
                                    image_sizes,
                                    in_format='cxcywh',
                                    out_format='xyxy',
                                    reverse=True)
        target_feat = self.get_target_roi_feat(search_feat, boxes_abs,
                                               self.mot_roi).view(
                                                   *det_boxes.shape[:2], -1)

        ref_info = {
            'proposal': det_boxes,
            'target_feat': target_feat,
            # 'feat': feature
        }
        out['pred_embs'] = target_feat

        return out, ref_info


@META_ARCH_REGISTRY.register()
class UnifiedTransformerTracker2(UnifiedTransformerTracker):
    ''''
  for MOT:
  reference detection are frozen for lower memory cost
  for sot:
  reference are random sampled
  '''

    def forward_sot(self, data: List[Dict[str, torch.Tensor]]):
        """for sot training
    Args:
        data (List[Dict[str, torch.Tensor]]): [description]

    """
        samples = self.preprocess_sot_inputs(data)
        batch, time = samples['search_images'].shape[:2]  ### B T C H W
        features = self.backbone(samples['search_images'].flatten(0, 1))
        feature_layer = self.sot_feature_layer
        srcs, masks, posembs = self.get_proj_features(
            features, samples['search_masks'].flatten(0, 1),
            self.tracking_proj)
        frame_feature = srcs[feature_layer]
        mask, posemb = masks[feature_layer], posembs[feature_layer]
        ### prepare tracking targets and proposals
        temp_indices = torch.randint(0, time, (batch, )) + torch.arange(
            batch) * time  ### random reference frame for each video
        temp_frame_feature = frame_feature[temp_indices]
        temp_boxes = samples['search_boxes'].view(-1, 4)[temp_indices]
        init_target_feat = self.get_target_roi_feat([temp_frame_feature],
                                                    temp_boxes, self.sot_roi)
        init_target_feat = repeat(init_target_feat, 'b c -> (b t) c',
                                  t=time).unsqueeze(
                                      1)  ## reference feature for all frames
        target_feat, proposals = self.prepare_sot_target_feature_proposal(
            init_target_feat, frame_feature, mask, posemb)

        #### track
        track_results = self.tracking_target(
            srcs, proposals, target_feat,
            samples['search_image_sizes'].flatten(0, 1))

        track_boxes = [proposals] + track_results['boxes']
        out = {
            "pred_boxes": track_boxes,
            "pred_scores": track_results.get("scores", None),
            "pred_masks": track_results.get("masks", None)
        }
        return self.sot_objective(out, samples)

    def forward_mot(self, data: List[Dict[str, torch.Tensor]]):
        """[summary]
    Args:
        data (List[Dict[str, torch.Tensor]]): [description]

    """

        samples = self.preprocess_mot_inputs(data)
        targets = self.prepare_mot_targets(data)
        # Extract backbone features
        with torch.no_grad():
            template_features = self.backbone(samples['template'].tensor)
            temp_detectoin = self.detect_objects(template_features,
                                                 samples['template'].mask)
            temp_boxes = temp_detectoin['boxes']
            temp_srcs, temp_masks, _ = self.get_proj_features(
                template_features, samples['template'].mask,
                self.tracking_proj)
            target_feat, proposals = self.prepare_mot_target_feature_proposal(
                temp_srcs, temp_boxes[-1], targets)
        search_features = self.backbone(samples['search'].tensor)
        det_out = self.detect_objects(search_features, samples['search'].mask)
        search_srcs, _, _ = self.get_proj_features(search_features,
                                                   samples['search'].mask,
                                                   self.tracking_proj)
        track_out = self.tracking_target(search_srcs,
                                         proposals,
                                         target_feat,
                                         targets['image_sizes'],
                                         box_format='cxcywh')
        out = {
            'pred_logits': det_out['logits'],
            'pred_boxes': det_out['boxes'],
            'track_boxes': track_out['boxes'],
            'track_scores': track_out['scores']
        }
        if self.mot_mask_emb is not None:
            out['pred_masks'] = det_out['masks']
        targets['det_targets'] = targets['track_targets']
        return self.mot_objective(out, targets)
