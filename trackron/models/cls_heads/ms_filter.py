import torch.nn as nn
import trackron.models.layers.filter as filter_layer
import torch
import math


class MultiScaleFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self):
        super().__init__()

        # self.feat_names = feat_names

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


        # Extract features
        train_feat = self.extract_classification_feat(train_feat)
        test_feat = self.extract_classification_feat(test_feat)

        # Train filter
        filter, filter_iter, levels = self.get_filter(train_feat, train_bb, *args, **kwargs)


        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat, levels) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat, levels):
        """Run classifier (filter) on the features (feat)."""
        num_sequence = weights.shape[0]
        feat = self.map_feat(feat, levels)
        feat= feat.view(-1, num_sequence, *feat.shape[-3:])

        scores = filter_layer.apply_filter(feat, weights)
        return scores
    
    def map_feat(self, features, levels):
        features = [feat for feat in features.values()]
        features = torch.stack([features[level][idx] for idx, level in enumerate(levels)])
        return features
        
    

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
        
        
        sequence = bb.shape[1] if len(bb) == 3 else 1
        ## to xyxy
        bbf = bb.reshape(-1, 4)
        bbf[:, 2:] = bbf[:, :2] + bbf[:, 2:]
        bbf = bbf.split(dim=0, split_size=1)
        weights, levels = self.filter_initializer(feat, bbf)
        feat = self.map_feat(feat, levels)
        weights = torch.mean(weights.view(-1, sequence, *weights.shape[-3:]), dim=0)
        feat = feat.view(-1, sequence, *feat.shape[-3:])

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, levels

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