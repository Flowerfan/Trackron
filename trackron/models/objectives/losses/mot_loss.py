import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from trackron.models.utils.warp_utils import flow_warp, get_occu_mask_bidirection, get_occu_mask_backward


class unFlowLoss(nn.modules.Module):
    def __init__(self, w_l1=0.15, w_ssim=0.85, w_ternary=0.0, w_smooth=50.0, occ_from_back=True, with_bk=True, smooth_2nd=True):
        super(unFlowLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_ternary = w_ternary
        self.w_smooth = w_smooth
        self.occ_from_back = occ_from_back
        self.with_bk = with_bk
        self.smooth_2nd=smooth_2nd

    def loss_photomatric(self, im1, im1_recons, occu_mask1):
        loss = []

        if self.w_l1 > 0:
            loss += [self.w_l1 *
                     (im1 - im1_recons).abs() * occu_mask1]

        if self.w_ssim > 0:
            loss += [self.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1 * occu_mask1)]

        if self.w_ternary > 0:
            loss += [self.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1 * occu_mask1)]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1):
        if self.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1, 10)]
        return sum([l.mean() for l in loss])

    def forward(self, output, im1_origin, im2_origin):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        seq_flows = output
        # im1_origin = target[:, :3]
        # im2_origin = target[:, 3:]

        seq_smooth_losses = []
        seq_warp_losses = []
        self.seq_occu_mask1 = []
        self.seq_occu_mask2 = []

        self.w_scales = [1.0 for _ in range(len(output))]
        self.w_sm_scales = [0.0 for _ in range(len(output))]
        self.w_sm_scales[0] = 1.0

        s = 1.
        for i, flow in enumerate(seq_flows):

            b, _, h, w = flow.size()

            im1_recons = flow_warp(
                im2_origin, flow[:, :2], pad="border")
            im2_recons = flow_warp(
                im1_origin, flow[:, 2:], pad="border")

            if self.occ_from_back:
                occu_mask1 = 1 - \
                    get_occu_mask_backward(flow[:, 2:], th=0.2)
                occu_mask2 = 1 - \
                    get_occu_mask_backward(flow[:, :2], th=0.2)
            else:
                occu_mask1 = 1 - \
                    get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                occu_mask2 = 1 - \
                    get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])

            self.seq_occu_mask1.append(occu_mask1)
            self.seq_occu_mask2.append(occu_mask2)

            loss_warp = self.loss_photomatric(
                im1_origin, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow[:, :2] / s, im1_origin)

            if self.with_bk:
                loss_warp += self.loss_photomatric(im2_origin, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow[:, 2:] / s, im2_origin)

                loss_warp /= 2.
                loss_smooth /= 2.

            seq_warp_losses.append(loss_warp)
            seq_smooth_losses.append(loss_smooth)

        seq_warp_losses = [l * w for l, w in
                               zip(seq_warp_losses, self.w_scales)]
        seq_smooth_losses = [l * w for l, w in
                                 zip(seq_smooth_losses, self.w_sm_scales)]

        warp_loss = sum(seq_warp_losses)
        smooth_loss = self.w_smooth * sum(seq_smooth_losses)
        total_loss = warp_loss + smooth_loss

        # return total_loss, warp_loss, smooth_loss, seq_flows[0].abs().mean()
        return total_loss
