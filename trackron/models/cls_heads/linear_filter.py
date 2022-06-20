import torch.nn as nn
from trackron.models.layers import filter as filter_layer
# import trackron.models.layers.filter as filter_layer
import torch
import math
from .build import CLS_HEAD_REGISTRY
from trackron.config import configurable
from .initializer import build_dimp_initializer
from .optimizer import build_dimp_optimizer
from .extractor import build_dimp_extractor


@CLS_HEAD_REGISTRY.register()
class DIMP_CLS_HEAD(nn.Module):

    @configurable
    def __init__(self, *, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None):
        """Target classification filter module.
        args:
            filter_size:  Size of filter (int).
            filter_initialize:  Filter initializer module.
            filter_optimizer:  Filter optimizer module.
            feature_extractor:  Feature extractor module applied to the input backbone features."""
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @classmethod
    def from_config(cls, cfg):
        filter_size = cfg.MODEL.DIMP
        filter_initializer = build_dimp_initializer(cfg)
        filter_optimizer = build_dimp_optimizer(cfg)
        feature_extractor = build_dimp_extractor(cfg)
        return {"filter_size": filter_size,
                "filter_initializer": filter_initializer,
                "filter_optimizer": filter_optimizer,
                "feature_extractor": feature_extractor}
        
    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb)

        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)
        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores


class GRUFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(
            train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        _, filter_iter = self.filter_optimizer(feat=train_feat, bb=train_bb, *args, **kwargs)

        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        weights, weights_iter = self.filter_optimizer(
            feat=feat, bb=bb, *args, **kwargs)

        return weights, weights_iter, 0

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1,
                                                  *backbone_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(
            backbone_feat, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1,
                                                  *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat = self.extract_classification_feat(
            backbone_feat, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores

class MixLinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, coef, rand_inds, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)
        mix_filters = [coef*f + (1-coef)*f[rand_inds] for f in filter_iter]
        # mix_feat = coef * test_feat + (1-coef) * test_feat[:, rand_inds, ...]

        # Classify samples using all return filters
        test_scores1 = [self.classify(f, test_feat) for f in mix_filters]
        test_scores2 = [self.classify(f, test_feat[:, rand_inds, ...]) for f in mix_filters]

        return test_scores1, test_scores2

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)
        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores


class MultiScaleFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_optimizer=None, feature_extractor=None, filter_norm=False):
        super().__init__()

        # self.feat_names = feat_names
        # self.filter_size = filter_size

        # Modules
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.filter_norm = filter_norm

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_temp, train_feat, test_feat, dist_map, num_seq, num_img):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""


        # if train_feat.dim() == 5:
        #     train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        # if test_feat.dim() == 5:
        #     test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        scale = 1.0 / torch.tensor(train_temp.shape[-3:], device=train_temp.device, dtype=train_temp.dtype).prod().sqrt()
        # Extract features
        train_feat, train_temp= self.normalize(train_feat, scale, train_temp)
        train_feat = train_feat.view(num_img, num_seq, *train_feat.shape[-3:])
        test_feat = self.normalize(test_feat, scale).view(num_img, num_seq, *test_feat.shape[-3:])
        
        # train_temp = self.extract_classification_feat(train_temp)
        # train_feat = self.extract_classification_feat(train_feat).view(num_img, num_seq, *train_feat.shape[-3:])
        # test_feat = self.extract_classification_feat(test_feat).view(num_img, num_seq, *test_feat.shape[-3:])
        train_temp = self.feature_extractor(train_temp)
        weights =  train_temp.view(num_img, num_seq, *train_temp.shape[-3:]).mean(0)
        # if self.filter_norm:
        #     weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])
        

        # Train filter
        filter, filter_iter, train_losses = self.get_filter(train_feat, weights, dist_map)


        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores
    
    def init_filter(self, feat, weights, dist_map, num_seq, num_img, *args, **kwargs):
        scale = 1.0 / torch.tensor(weights.shape[-3:], device=weights.device, dtype=weights.dtype).prod().sqrt()
        num_sequence = weights.shape[0]
        feat, weights = self.normalize(feat, scale, weights)
        weights = self.extract_classification_feat(weights)
        weights =  weights.view(num_img, num_seq, *weights.shape[-3:]).mean(0)
        filter, filter_iter, train_losses = self.get_filter(feat, weights, dist_map)
        # feat = self.extract_classification_feat(feat.view(-1, *feat.shape[-3:])).view(num_img, num_seq, *feat.shape[-3:])
        return filter 
        # filter, filter_iter, levels = self.get_filter(feat, weights, dist_map)
        # return filter

    def normalize(self, input, scale, weights=None):
        scale_factor = (scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                    torch.sum((input * input).contiguous().view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + 1e-5)).sqrt())
        if weights is None:
            return input * scale_factor
        else:
            return input * scale_factor, weights * scale_factor.detach()

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""
        scale = 1.0 / torch.tensor(weights.shape[-3:], device=weights.device, dtype=weights.dtype).prod().sqrt()
        num_sequence = weights.shape[0]
        feat = self.normalize(feat, scale)
        feat= feat.view(-1, num_sequence, *feat.shape[-3:])

        scores = filter_layer.apply_filter(feat, weights)
        return scores
    

    def get_filter(self, feat, weights, dist_map, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        
        
        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, dist_map=dist_map, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores


class MultiScaleFilter2(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_optimizer=None, filter_op=None):
        super().__init__()

        # self.feat_names = feat_names
        # self.filter_size = filter_size

        # Modules
        self.filter_optimizer = filter_optimizer
        self.filter_op = filter_op

        # Init weights
        for m in self.filter_op.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_temp, train_feat, test_feat, dist_map, num_seq, num_img):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""



        # Extract features
        train_feat = train_feat.view(num_img, num_seq, *train_feat.shape[-3:])
        test_feat = test_feat.view(num_img, num_seq, *test_feat.shape[-3:])
        
        # train_temp = self.extract_classification_feat(train_temp)
        # train_feat = self.extract_classification_feat(train_feat).view(num_img, num_seq, *train_feat.shape[-3:])
        # test_feat = self.extract_classification_feat(test_feat).view(num_img, num_seq, *test_feat.shape[-3:])
        train_temp = self.filter_op(train_temp)
        weights =  train_temp.view(num_img, num_seq, *train_temp.shape[-3:]).mean(0)
        # if self.filter_norm:
        #     weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])
        

        # Train filter
        filter, filter_iter, train_losses = self.get_filter(train_feat, weights, dist_map)


        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores
    
    def init_filter(self, feat, weights, dist_map, num_seq, num_img, *args, **kwargs):
        num_sequence = weights.shape[0]
        weights = self.filter_op(weights)
        weights =  weights.view(num_img, num_seq, *weights.shape[-3:]).mean(0)
        filter, filter_iter, train_losses = self.get_filter(feat, weights, dist_map)
        # feat = self.extract_classification_feat(feat.view(-1, *feat.shape[-3:])).view(num_img, num_seq, *feat.shape[-3:])
        return filter 
        # filter, filter_iter, levels = self.get_filter(feat, weights, dist_map)
        # return filter

    # def extract_classification_feat(self, feat, num_sequences=None):
    #     """Extract classification features based on the input backbone features."""
    #     if self.feature_extractor is None:
    #         return feat
    #     if num_sequences is None:
    #         return self.feature_extractor(feat)
    #     output = self.feature_extractor(feat)
    #     return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""
        num_sequence = weights.shape[0]
        feat= feat.view(-1, num_sequence, *feat.shape[-3:])
        scores = filter_layer.apply_filter(feat, weights)
        return scores
    

    def get_filter(self, feat, weights, dist_map, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        
        
        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, dist_map=dist_map, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses


class ContraLinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, neck, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.neck = neck

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)

        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]
        filter_feats = torch.stack([filter_iter[0], filter_iter[-1]], dim=0).reshape(num_sequences*2, *filter.shape[-3:])
        filter_feats = self.neck(filter_feats).view(2, num_sequences, -1)

        return test_scores, filter_feats

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)
        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

class ContraTrainTestLinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, neck, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.neck = neck

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, test_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        train_filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)
        test_filter = self.filter_initializer(test_feat, test_bb)

        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]
        filter_feats = torch.stack([train_filter, test_filter], dim=0).reshape(num_sequences*2, *train_filter.shape[-3:])
        filter_feats = self.neck(filter_feats).view(2, num_sequences, -1)

        return test_scores, filter_feats

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)
        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses




class TrLinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None, transformer=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

        self.transformer = transformer

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, train_label, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]
        num_img_test = test_feat.shape[0] 

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # 
        encoded_feat, decoded_feat = self.transformer(train_feat, test_feat, train_label)

        encoded_feat = encoded_feat.reshape(-1, num_sequences, *encoded_feat.shape[-3:])
        decoded_feat = decoded_feat.reshape(-1, num_sequences, *decoded_feat.shape[-3:])

        # Train filter
        filter, filter_iter, losses = self.get_filter(encoded_feat, train_bb, *args, **kwargs)   
        # Classify samples using all return filters
        test_scores = [self.classify(f, decoded_feat) for f in filter_iter]    ## test_feat

        '''
        # Train filter
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs) 
        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]    ## test_feat
        '''

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores
