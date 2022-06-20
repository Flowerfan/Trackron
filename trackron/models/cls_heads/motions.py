
import torch
import torch.nn as nn
import torch.nn.function as F
import trackron.models.layers.filter as filter_layer
import math

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
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


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim,
                                hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, opt):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = opt.corr_levels * (2*opt.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    def __init__(self, opt):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = opt.corr_levels * (2*opt.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=1, radius=3):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []

        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx),
                                axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


def upflow2(flow, mode='bilinear'):
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return 2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CorrFlow(nn.Module):
    def __init__(self, opt, hdim=96):
        super(CorrFlow, self).__init__()

        self.opt = opt
        self.radius = 3
        self.iters = 3

        # flow update block
        self.update_block = SmallUpdateBlock(opt, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, fmap1, fmap2, cmap, cinp, corr_fn):
        """ Estimate optical flow between pair of feats """

        coords0, coords1 = self.initialize_flow(fmap1)
        # if flow_init is not None:
        #     coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(self.iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.opt.mixed_precision):
                cmap, up_mask, delta_flow = self.update_block(
                    cmap, cinp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            flow_up = upflow2(coords1 - coords0)

            flow_predictions.append(flow_up)

        return cmap, flow_predictions




class CoFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None, corr_levels=1, corr_radius=3):
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

        # visual filter
        v_filter, v_filter_iter, losses = self.get_appearence_filter(train_feat, train_bb, *args, **kwargs)
        

        # Classify samples using visual filters
        test_scores = [self.classify(f, test_feat) for f in v_filter_iter]

        # Motion filter
        corr_fn = CorrBlock(
            train_feat, test_feat, num_levels=self.corr_levels, radius=self.corr_radius)
        test_motion_feat = self.extract_motion_feat(train_feat, test_feat)

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

    def get_appearence_filter(self, feat, bb, *args, **kwargs):
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
