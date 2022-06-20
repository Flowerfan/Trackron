import torch.nn as nn
import torch
import torch.nn.functional as F
import trackron.models.layers.filter as filter_layer
from .initializer import FilterPool
from trackron.models.layers.blocks import conv_block
import trackron.models.layers.activation as activation
from trackron.models.layers.distance import DistanceMap
import math


class FilterHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, out_dim=256):
        super(FilterHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ContextHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, context_dim=128):
        super(ContextHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim+context_dim, 4, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2(self.relu(self.conv1(x)))
        return torch.split(x, [self.hidden_dim, self.context_dim], dim=1)
        

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class CostEncoder(nn.Module):
    def __init__(self, input_dim=256, cost_dim=127):
        super(CostEncoder, self).__init__()
        self.convc1 = nn.Conv2d(input_dim, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 128, 4, padding=2)
        self.convf1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+128, cost_dim, 3, padding=1)

    def forward(self, cost, feat):
        feat = F.relu(self.convc1(feat))
        feat = F.relu(self.convc2(feat))
        cos = F.relu(self.convf1(cost))
        cos = F.relu(self.convf2(cos))
        cost_feat = torch.cat([cos, feat], dim=1)
        out = F.relu(self.conv(cost_feat))
        return torch.cat([out, cost], dim=1)


class FilterUpdate(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(FilterUpdate, self).__init__()
        self.encoder = CostEncoder(input_dim, hidden_dim-1)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim+hidden_dim)
        self.filter_head = FilterHead(hidden_dim=128, out_dim=input_dim)

    def forward(self, filter_map, context_map, feat, cost):
        cost_feat = self.encoder(cost, feat)
        context_map = torch.cat([context_map, cost_feat], dim=1)
        filter_map = self.gru(filter_map, context_map)
        delta_filter = self.filter_head(filter_map)

        return filter_map, delta_filter


class GRUFilter(nn.Module):
    """Optimizer module for DiMP.
    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.
    Moreover it learns parameters in the loss itself, as described in the DiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        init_gauss_sigma:  The standard deviation to use for the initialization of the label function.
        num_dist_bins:  Number of distance bins used for learning the loss label, mask and weight.
        bin_displacement:  The displacement of the bins (level of discritization).
        mask_init_factor:  Parameter controlling the initialization of the target mask.
        score_act:  Type of score activation (target mask computation) to use. The default 'relu' is what is described in the paper.
        act_param:  Parameter for the score_act.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        mask_act:  What activation to do on the output of the mask computation ('sigmoid' or 'linear').
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """

    def __init__(self, filter_pool, num_iter=1, feat_stride=16, init_step_length=1.0, 
                 init_filter_reg=1e-2, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0,
                 score_act='relu', act_param=None, min_filter_reg=1e-3, mask_act='sigmoid',
                 num_filter_pre_convs=1, filter_size=1, feature_dim=256, hidden_dim=128, context_dim=128):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(
            math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        # Make pre conv
        # pre_conv_layers = []
        # for i in range(num_filter_pre_convs):
        #     pre_conv_layers.append(conv_block(
        #         feature_dim, feature_dim, kernel_size=filter_size, padding=2))
        # self.filter_pre_layers = nn.Sequential(
        #     *pre_conv_layers) if pre_conv_layers else None
        self.filter_encoder = FilterHead(feature_dim, hidden_dim, feature_dim)
        self.context_encoder = ContextHead(
            feature_dim, hidden_dim, context_dim)
        self.filter_pool = filter_pool
        self.filter_update = FilterUpdate(feature_dim, hidden_dim)

        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(
            1, -1, 1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        # Module that predicts the target label function (y in the paper)
        self.label_map_predictor = nn.Conv2d(
            num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        # Module that predicts the target mask (m in the paper)
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * \
            torch.tanh(2.0 - d) + init_bias

        # Module that predicts the residual weights (v in the paper)
        self.spatial_weight_predictor = nn.Conv2d(
            num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        # The score actvation and its derivative
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(
                act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown score activation')

    def forward(self, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        
        ## get init weights
        init_feat = feat.reshape(-1,
                                 feat.shape[-3], feat.shape[-2], feat.shape[-1])
        filter_feat = self.filter_encoder(init_feat)
        h_map, context_map = self.context_encoder(init_feat)
        weights = self.filter_pool(filter_feat, bb)
        if feat.shape[0] > 1:
            weights = torch.mean(weights.reshape(
                feat.shape[0], -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)

        
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute distance map
        dmap_offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).flip((-1,)) - dmap_offset
        dist_map = self.distance_map(center, output_sz)

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        target_mask = self.target_mask_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])

        # Get total sample weights
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(
                num_images, num_sequences, 1, 1) * spatial_weight

    

        weight_iterates = [weights]

        for i in range(num_iter):

            # Compute residuals
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = sample_weight * (scores_act - label_map)
            residuals_mapped = score_mask *  (sample_weight * residuals)
            residuals_mapped = residuals_mapped.reshape(-1, 1, residuals_mapped.shape[-2], residuals_mapped.shape[-1])

            # Compute gradient
            h_map, delta_filter = self.filter_update(h_map, context_map, init_feat, residuals_mapped)
            delta_weights = self.filter_pool(delta_filter, bb)
            if num_images > 1:
                delta_weights = torch.mean(delta_weights.reshape(
                    num_images, -1, delta_weights.shape[-3], delta_weights.shape[-2], delta_weights.shape[-1]), dim=0)

            weights = weights + delta_weights

            # Add the weight iterate
            weight_iterates.append(weights)

        return weights, weight_iterates







class ConvGRUFilter(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=256):
        super(ConvGRUFilter, self).__init__()
        self.convz = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convf = nn.Conv2d(hidden_dim, hidden_dim, 6, stride=4, padding=0)
        self.convq = nn.Conv2d(hidden_dim+hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, weights, feat, residual):
        # import pdb;pdb.set_trace()
        hx = filter_layer.apply_feat_transpose(
            feat, residual, weights.shape[-2:], training=self.training)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        # hf = torch.mean(self.convf(feat.reshape(-1, *feat.shape[2:])).reshape(*feat.shape[:2], *weights.shape[-3:]), dim=0)
        delta_weights = torch.tanh(self.convq(torch.cat([r*weights, hx], dim=1)))
        new_weights = (1-z) * weights + z * delta_weights
        return new_weights


class GRUFilterComp(nn.Module):
    """Optimizer module for DiMP.
    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.
    Moreover it learns parameters in the loss itself, as described in the DiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        init_gauss_sigma:  The standard deviation to use for the initialization of the label function.
        num_dist_bins:  Number of distance bins used for learning the loss label, mask and weight.
        bin_displacement:  The displacement of the bins (level of discritization).
        mask_init_factor:  Parameter controlling the initialization of the target mask.
        score_act:  Type of score activation (target mask computation) to use. The default 'relu' is what is described in the paper.
        act_param:  Parameter for the score_act.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        mask_act:  What activation to do on the output of the mask computation ('sigmoid' or 'linear').
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """

    def __init__(self, filter_pool, num_iter=1, feat_stride=16, init_step_length=1.0,
                 init_filter_reg=1e-2, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0,
                 score_act='relu', act_param=None, min_filter_reg=1e-3, mask_act='sigmoid',
                 num_filter_pre_convs=1, filter_size=1, feature_dim=256, hidden_dim=128, context_dim=128):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(
            math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        self.filter_encoder = FilterHead(feature_dim, hidden_dim, feature_dim)
        self.filter_pool = filter_pool
        self.filter_update = ConvGRUFilter(feature_dim, hidden_dim)

        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(
            1, -1, 1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        # Module that predicts the target label function (y in the paper)
        self.label_map_predictor = nn.Conv2d(
            num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        # Module that predicts the target mask (m in the paper)
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * \
            torch.tanh(2.0 - d) + init_bias

        # Module that predicts the residual weights (v in the paper)
        self.spatial_weight_predictor = nn.Conv2d(
            num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        # The score actvation and its derivative
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(
                act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown score activation')

    def forward(self, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        ## get init weights
        init_feat = feat.reshape(-1,
                                 feat.shape[-3], feat.shape[-2], feat.shape[-1])
        filter_feat = self.filter_encoder(init_feat)
        weights = self.filter_pool(filter_feat, bb)
        if feat.shape[0] > 1:
            weights = torch.mean(weights.reshape(
                feat.shape[0], -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)

        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) %
                     2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg *
                      self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute distance map
        dmap_offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) /
                  self.feat_stride).flip((-1,)) - dmap_offset
        dist_map = self.distance_map(center, output_sz)

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:])
        target_mask = self.target_mask_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:])

        # Get total sample weights
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(
                num_images, num_sequences, 1, 1) * spatial_weight

        weight_iterates = [weights]

        for i in range(num_iter):
            # Compute residuals
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = sample_weight * (scores_act - label_map)
            residuals_mapped = score_mask * (sample_weight * residuals)
            residuals_mapped = residuals_mapped.reshape(
                -1, 1, residuals_mapped.shape[-2], residuals_mapped.shape[-1])

            # Compute gradient
            weights = self.filter_update(weights, feat, residuals_mapped)

            # weights = weights

            # Add the weight iterate
            weight_iterates.append(weights)

        return weights, weight_iterates


class RSM(nn.Module):
    """Optimizer module for DiMP.
    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.
    Moreover it learns parameters in the loss itself, as described in the DiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        init_gauss_sigma:  The standard deviation to use for the initialization of the label function.
        num_dist_bins:  Number of distance bins used for learning the loss label, mask and weight.
        bin_displacement:  The displacement of the bins (level of discritization).
        mask_init_factor:  Parameter controlling the initialization of the target mask.
        score_act:  Type of score activation (target mask computation) to use. The default 'relu' is what is described in the paper.
        act_param:  Parameter for the score_act.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        mask_act:  What activation to do on the output of the mask computation ('sigmoid' or 'linear').
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """

    def __init__(self, filter_pool, num_iter=1, feat_stride=16, init_step_length=1.0,
                 init_filter_reg=1e-2, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0,
                 score_act='relu', act_param=None, min_filter_reg=1e-3, mask_act='sigmoid',
                 num_filter_pre_convs=1, filter_size=1, feature_dim=256, hidden_dim=128, context_dim=128):
        super().__init__()

        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(
            math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        # Make pre conv
        # pre_conv_layers = []
        # for i in range(num_filter_pre_convs):
        #     pre_conv_layers.append(conv_block(
        #         feature_dim, feature_dim, kernel_size=filter_size, padding=2))
        # self.filter_pre_layers = nn.Sequential(
        #     *pre_conv_layers) if pre_conv_layers else None
        self.filter_encoder = FilterHead(feature_dim, hidden_dim, feature_dim)
        self.context_encoder = ContextHead(
            feature_dim, hidden_dim, context_dim)
        self.filter_pool = filter_pool
        self.filter_update = FilterUpdate(feature_dim, hidden_dim)

        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(
            1, -1, 1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        # Module that predicts the target label function (y in the paper)
        self.label_map_predictor = nn.Conv2d(
            num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        # Module that predicts the target mask (m in the paper)
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * \
            torch.tanh(2.0 - d) + init_bias

        # Module that predicts the residual weights (v in the paper)
        self.spatial_weight_predictor = nn.Conv2d(
            num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        # The score actvation and its derivative
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(
                act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown score activation')


    def init_score_map(self, feat, bb):
        
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        output_sz = (feat.shape[-2] + (weights.shape[-2] + 1) %
                     2, feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg *
                      self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute distance map
        dmap_offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) /
                  self.feat_stride).flip((-1,)) - dmap_offset
        dist_map = self.distance_map(center, output_sz)

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:])
        target_mask = self.target_mask_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(
            num_images, num_sequences, *dist_map.shape[-2:])

        # Get total sample weights
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(
                num_images, num_sequences, 1, 1) * spatial_weight

    def forward(self, train_feat, test_feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        # Sizes
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        ## get init weights
        init_feat = feat.reshape(-1,
                                 feat.shape[-3], feat.shape[-2], feat.shape[-1])
        filter_feat = self.filter_encoder(init_feat)
        h_map, context_map = self.context_encoder(init_feat)
        weights = self.filter_pool(filter_feat, bb)
        if feat.shape[0] > 1:
            weights = torch.mean(weights.reshape(
                feat.shape[0], -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)

        filter_sz = (weights.shape[-2], weights.shape[-1])
        

        weight_iterates = [weights]
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        for i in range(num_iter):

            # Compute residuals
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = sample_weight * (scores_act - label_map)
            residuals_mapped = score_mask * (sample_weight * residuals)
            residuals_mapped = residuals_mapped.reshape(
                -1, 1, residuals_mapped.shape[-2], residuals_mapped.shape[-1])

            # Compute gradient
            h_map, delta_filter = self.filter_update(
                h_map, context_map, init_feat, residuals_mapped)
            delta_weights = self.filter_pool(delta_filter, bb)
            if num_images > 1:
                delta_weights = torch.mean(delta_weights.reshape(
                    num_images, -1, delta_weights.shape[-3], delta_weights.shape[-2], delta_weights.shape[-1]), dim=0)

            weights = weights + delta_weights

            # Add the weight iterate
            weight_iterates.append(weights)

        return weights, weight_iterates
