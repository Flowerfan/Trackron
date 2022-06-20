import torch
import torch.nn as nn
from collections import defaultdict
from torchvision.ops import box_convert
from einops import rearrange
from trackron.config import configurable
from trackron.models.poolers import ROIPooler
from trackron.models.layers.activation import get_activation_fn
from trackron.utils.misc import clone_modules
from trackron.structures import Boxes
from trackron.models.mask_heads import build_mask_head
from timm.models.layers import DropPath

from .box_regression import Box2BoxTransform
from .build import BOX_HEAD_REGISTRY


@BOX_HEAD_REGISTRY.register()
class TargetTransformer(nn.Module):

    @configurable
    def __init__(
        self,
        iterations,
        pooler,
        mask_pooler,
        interact_layers,
        box_module,
        mask_module,
        score_module,
        box_transform,
        output_score,
        expand_scale=1.0,
    ):
        super().__init__()

        self.iterations = iterations
        self.interact_layers = interact_layers
        self.pooler = pooler
        self.mask_pooler = mask_pooler
        self.box_module = box_module
        self.mask_module = mask_module
        self.score_module = score_module
        self.box_transform = box_transform
        self.output_score = output_score
        self.expand_scale = expand_scale
        self.init_parameters()

    @classmethod
    def from_config(cls, cfg):
        iterations = cfg.ITERATIONS
        hidden_dim = cfg.FEATURE_DIM
        dim_feedforward = cfg.DIM_FEEDFORWARD
        dropout = cfg.DROPOUT
        pooler_resolution = cfg.POOL_SIZE
        pooler_scales = cfg.POOL_SCALES
        sampling_ratio = cfg.POOL_SAMPLE_RATIO
        pooler_type = cfg.POOL_TYPE
        activation = cfg.ACTIVATION
        num_box_layer = cfg.NUM_BOX_LAYER
        box_weights = cfg.BOX_WEIGHTS
        output_score = cfg.OUTPUT_SCORE
        expand_scale = cfg.EXPAND_SCALE

        pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        interact = Correlation(hidden_dim, pooler_resolution)
        interact_layer = InteractLayer(hidden_dim,
                                       interact,
                                       dim_feedforward,
                                       dropout=dropout,
                                       activation=activation)
        box_layer = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim, False),
                              nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True))
                for _ in range(num_box_layer)
            ], nn.Linear(hidden_dim, 4))
        mask_pooler = mask_module = None
        if cfg.WITH_MASK:
            mask_pooler = ROIPooler(
                output_size=cfg.MASK_HEAD.POOL_SIZE,
                scales=cfg.MASK_HEAD.POOL_SCALES,
                sampling_ratio=cfg.MASK_HEAD.POOL_SAMPLE_RATIO,
                pooler_type=cfg.MASK_HEAD.POOL_TYPE,
            )
            mask_head = build_mask_head(cfg.MASK_HEAD)
            mask_module = clone_modules(mask_head, iterations)
        interact_layers = clone_modules(interact_layer, iterations)
        box_module = clone_modules(box_layer, iterations)
        box_transform = Box2BoxTransform(box_weights)
        score_module = None
        if output_score:
            score_layer = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim, False),
                                  nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True))
                    for _ in range(num_box_layer)
                ], nn.Linear(hidden_dim, 1))
            score_module = clone_modules(score_layer, iterations)
        return {
            "iterations": iterations,
            "interact_layers": interact_layers,
            "pooler": pooler,
            "mask_pooler": mask_pooler,
            "box_module": box_module,
            "mask_module": mask_module,
            "score_module": score_module,
            "box_transform": box_transform,
            "output_score": output_score,
            "expand_scale": expand_scale
        }

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, obj_features, boxes, pooler=None, obj_masks=None):
        """[summary]

    Args:
        features ([torch.Tensor]): [B C H W]
        obj_features ([torch.Tensor]): [B N C]
        boxes ([torch.Tensor]): [B N 4]

    Returns:
        [type]: [description]
    """

        B, N = boxes.shape[:2]
        results = defaultdict(list)
        pooler = pooler if pooler is not None else self.pooler
        if not isinstance(features, (list, tuple)):
            features = [features]
        for it in range(self.iterations):
            proposal_boxes = self._get_proposal_bboxes(boxes)
            roi_features = pooler(features, proposal_boxes)
            roi_features = rearrange(roi_features, 'BN C H W -> BN (H W) C')

            ### refine obj_feature and box coordinates
            obj_features = self.interact_layers[it](roi_features,
                                                    obj_features,
                                                    key_padding_mask=obj_masks)
            boxes_deltas = self.box_module[it](obj_features)
            boxes = self.post_process_boxes(boxes, boxes_deltas)

            results['boxes'].append(boxes)
            results['features'].append(obj_features)
            if self.mask_module is not None:
                roi_features = self.mask_pooler(features, proposal_boxes)
                target_mask = self.mask_module[it](obj_features.flatten(0, 1),
                                                   roi_features)
                target_mask = rearrange(target_mask, '(B N) C H W -> B N C H W', B=B)
                results['masks'].append(target_mask)
            if self.output_score:
                scores = self.score_module[it](obj_features).squeeze(-1)
                results['scores'].append(scores)
        return results

    def _get_proposal_bboxes(self, boxes):
        if self.expand_scale != 1.0:
            width = torch.clamp_min(boxes[..., 2] - boxes[..., 0], 1.0)
            height = torch.clamp_min(boxes[..., 3] - boxes[..., 1], 1.0)
            # expand_width = width * (self.expand_scale - 1) / 2.0
            # expand_height = height * (self.expand_scale - 1) / 2.0
            # boxes = boxes + torch.stack([-expand_width, -expand_height, expand_width, expand_height], dim=-1)
            ### expand to squre search area
            new_size = (width * height * self.expand_scale).sqrt().detach()
            expand_width = (new_size - width) / 2.0
            expand_height = (new_size - height) / 2.0
            boxes = boxes + torch.stack([-expand_width, -expand_height, expand_width, expand_height], dim=-1)

        proposal_boxes = [Boxes(bbs) for bbs in boxes]
        return proposal_boxes


    def post_process_boxes(self, boxes, boxes_deltas):
        input_shape = boxes.shape
        boxes = box_convert(boxes.view(-1, 4), 'xyxy', 'xywh')
        pred_bboxes = self.box_transform.apply_deltas(boxes_deltas.view(-1, 4),
                                                        boxes)
        boxes = box_convert(pred_bboxes, 'xywh', 'xyxy')
        return boxes.view(input_shape)

class InteractLayer(nn.Module):

    def __init__(
        self,
        d_model,
        interact,
        dim_feedforward=2048,
        nhead=8,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.interact = interact

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = DropPath(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = DropPath(dropout)
        self.dropout2 = DropPath(dropout)
        self.dropout3 = DropPath(dropout)

        self.activation = get_activation_fn(activation)

    def forward(self, roi_features, obj_features, key_padding_mask=None):
        """[summary]

    Args:
        roi_features ([tensor]): [BN K C] K = HxW
        obj_features ([tensor]): [B N C]

    Returns:
        [type]: [description]
    """
        input_shape = obj_features.shape
        # proposal features
        obj_features = rearrange(obj_features, 'B N C -> N B C')
        obj_features2 = self.self_attn(obj_features,
                                       obj_features,
                                       value=obj_features,
                                       key_padding_mask=key_padding_mask)[0]
        obj_features = obj_features + self.dropout1(obj_features2)
        obj_features = self.norm1(obj_features)

        ### interact
        obj_features = rearrange(obj_features, 'N B C -> (B N) C')
        obj_features2 = self.interact(obj_features.unsqueeze(1), roi_features)
        obj_features = obj_features + self.dropout2(obj_features2)
        obj_features = self.norm2(obj_features)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        return obj_features.reshape(*input_shape)



class Correlation(nn.Module):

    def __init__(self, hidden_dim, pooler_resolution):
        super().__init__()

        self.hidden_dim = hidden_dim
        num_output = self.hidden_dim * pooler_resolution**2
        self.fc1 = nn.Linear(num_output, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim * 2)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)


    def forward(self, obj_features, roi_features):
        '''
        obj_features: (N, 1, C)
        roi_features: (N, 49, C)
        '''
        features = torch.einsum('nqc,nkc->nkq', obj_features, roi_features)
        features = features * roi_features
        features = self.fc1(features.flatten(1))
        features = self.norm1(features)
        features = self.activation(features)

        features = self.fc2(features)
        features = self.norm2(features)
        features = self.activation(features)

        return features
