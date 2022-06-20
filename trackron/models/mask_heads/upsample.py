import torch
import torch.nn as nn


if torch.__version__ == 'parrots':
  TORCH_VERSION = torch.__version__
else:
  # torch.__version__ could be 1.3.1+cu92, we only need the first two
  # for comparison
  TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


def obsolete_torch_version(torch_version, version_threshold):
  return torch_version == 'parrots' or torch_version <= version_threshold


class NewEmptyTensorOp(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, new_shape):
    ctx.shape = x.shape
    return x.new_empty(new_shape)

  @staticmethod
  def backward(ctx, grad):
    shape = ctx.shape
    return NewEmptyTensorOp.apply(grad, shape), None


class ConvTranspose2d(nn.ConvTranspose2d):

  def forward(self, x):
    if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
      out_shape = [x.shape[0], self.out_channels]
      for i, k, p, s, d, op in zip(x.shape[-2:], self.kernel_size, self.padding,
                                   self.stride, self.dilation,
                                   self.output_padding):
        out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
      empty = NewEmptyTensorOp.apply(x, out_shape)
      if self.training:
        # produce dummy gradient to avoid DDP warning.
        dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
        return empty + dummy
      else:
        return empty

    return super().forward(x)