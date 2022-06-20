from typing import Union

import torch
from torch import nn, Tensor

# from torchvision.ops import roi_align
from trackron.external import prroi_pool2d as roi_align

from torchvision.ops.boxes import box_area
from torch.jit.annotations import Optional, List, Dict, Tuple
import torchvision


# copying result_idx_in_level to a specific index in result[]
# is not supported by ONNX tracing yet.
# _onnx_merge_levels() is an implementation supported by ONNX
# that merges the levels to the right indices
@torch.jit.unused
def _onnx_merge_levels(levels: Tensor, unmerged_results: List[Tensor]) -> Tensor:
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros((levels.size(0), first_result.size(1),
                       first_result.size(2), first_result.size(3)),
                      dtype=dtype, device=device)
    for level in range(len(unmerged_results)):
        index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        index = index.expand(index.size(0),
                             unmerged_results[level].size(1),
                             unmerged_results[level].size(2),
                             unmerged_results[level].size(3))
        res = res.scatter(0, index, unmerged_results[level])
    return res


# TODO: (eellison) T54974082 https://github.com/pytorch/pytorch/issues/26744/pytorch/issues/26744
def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 224,
    canonical_level: int = 4,
    eps: float = 1e-6,
):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Arguments:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(
            self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics present in the FPN paper.

    Arguments:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign

    Examples::

        >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
        >>> i = OrderedDict()
        >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        >>> # create some random bounding boxes
        >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        >>> # original image size, before computing the feature maps
        >>> image_sizes = [(512, 512)]
        >>> output = m(i, [boxes], image_sizes)
        >>> print(output.shape)
        >>> torch.Size([6, 5, 3, 3])

    """

    __annotations__ = {
        'scales': Optional[List[float]],
        'map_levels': Optional[LevelMapper]
    }

    def __init__(
        self,
        featmap_names: List[str],
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
        scales: List[float],
        map_levels: initLevelMapper,
        feature_dim=256,
        norm_layer=None
    ):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = scales
        self.map_levels = map_levels
        self.norm_layer = norm_layer
        # self.filter_convs = nn.ModuleList([nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1) for _ in scales])
        # lvl_min = - \
        #     torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        # lvl_max = - \
        #     torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        # self.scales = scales
        # self.map_levels = initLevelMapper(int(lvl_min), int(lvl_max))

    def convert_to_roi_format(self, boxes: List[Tensor]) -> Tensor:
        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat(
            [
                torch.full_like(b[:, :1], i, dtype=dtype,
                                layout=torch.strided, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        output_sz: Optional[Tuple] = None,
        use_norm: Optional[bool] = False
    ) -> Tensor:
        """
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_names:
                x_filtered.append(v)
        num_levels = len(x_filtered)
        rois = self.convert_to_roi_format(boxes)
        scales = self.scales
        if output_sz is None:
            output_sz = self.output_size

        if num_levels == 1:
            return roi_align(
                x_filtered[0], rois,
                output_size=output_sz,
                spatial_scale=scales[0],
                sampling_ratio=self.sampling_ratio
            )

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x_filtered[0].shape[1]

        dtype, device = x_filtered[0].dtype, x_filtered[0].device
        result = torch.zeros(
            (num_rois, num_channels,) + output_sz,
            dtype=dtype,
            device=device,
        )

        tracing_results = []
        for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
            idx_in_level = torch.where(levels == level)[0]
            # feature = self.filter_convs[level](per_level_feature)
            rois_per_level = rois[idx_in_level]
            result_idx_in_level = roi_align(
                per_level_feature, rois_per_level,
                output_sz[0], output_sz[1], scale)

            # result_idx_in_level = roi_align(
            #     per_level_feature, rois_per_level,
            #     output_size=output_sz,
            #     spatial_scale=scale, sampling_ratio=self.sampling_ratio)

            if torchvision._is_tracing():
                tracing_results.append(result_idx_in_level.to(dtype))
            else:
                # result and result_idx_in_level's dtypes are based on dtypes of different
                # elements in x_filtered.  x_filtered contains tensors output by different
                # layers.  When autocast is active, it may choose different dtypes for
                # different layers' outputs.  Therefore, we defensively match result's dtype
                # before copying elements from result_idx_in_level in the following op.
                # We need to cast manually (can't rely on autocast to cast for us) because
                # the op acts on result in-place, and autocast only affects out-of-place ops.
                result[idx_in_level] = result_idx_in_level.to(result.dtype)

        if torchvision._is_tracing():
            result = _onnx_merge_levels(levels, tracing_results)

        return result

    def level_pool(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        level: int,
        output_sz: Optional[Tuple] = None
    ) -> Tensor:
        """
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        assert 0 <= level <= len(x)
        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_names:
                x_filtered.append(v)
        rois = self.convert_to_roi_format(boxes)
        scales = self.scales
        if output_sz is None:
            output_sz = self.output_size

        return roi_align(
            x_filtered[level], rois,
            output_size=output_sz,
            spatial_scale=scales[level],
            sampling_ratio=self.sampling_ratio
        )
    def multi_level_pool(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        levels: Tensor,
        output_sz: Optional[Tuple] = None
    ) -> Tensor:
        """
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_names:
                x_filtered.append(v)
        rois = self.convert_to_roi_format(boxes)
        scales = self.scales
        if output_sz is None:
            output_sz = self.output_size

        num_rois = len(rois)
        num_channels = x_filtered[0].shape[1]

        dtype, device = x_filtered[0].dtype, x_filtered[0].device
        result = torch.zeros(
            (num_rois, num_channels,) + output_sz,
            dtype=dtype,
            device=device,
        )

        tracing_results = []
        for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
            idx_in_level = torch.where(levels == level)[0]
            # feature = self.filter_convs[level](per_level_feature)
            rois_per_level = rois[idx_in_level]
            result_idx_in_level = roi_align(
                per_level_feature, rois_per_level,
                output_sz[0], output_sz[1], scale)

            # result_idx_in_level = roi_align(
            #     per_level_feature, rois_per_level,
            #     output_size=output_sz,
            #     spatial_scale=scale, sampling_ratio=self.sampling_ratio)

            if torchvision._is_tracing():
                tracing_results.append(result_idx_in_level.to(dtype))
            else:
                # result and result_idx_in_level's dtypes are based on dtypes of different
                # elements in x_filtered.  x_filtered contains tensors output by different
                # layers.  When autocast is active, it may choose different dtypes for
                # different layers' outputs.  Therefore, we defensively match result's dtype
                # before copying elements from result_idx_in_level in the following op.
                # We need to cast manually (can't rely on autocast to cast for us) because
                # the op acts on result in-place, and autocast only affects out-of-place ops.
                result[idx_in_level] = result_idx_in_level.to(result.dtype)

        if torchvision._is_tracing():
            result = _onnx_merge_levels(levels, tracing_results)

        return result

