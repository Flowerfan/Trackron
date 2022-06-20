from collections import OrderedDict
import torch
import copy
import functools
import torchvision
from torch import Tensor
from typing import Optional, List, Tuple


class TensorDict(OrderedDict):
  """Container mainly used for dicts of torch tensors. Extends OrderedDict with pytorch functionality."""

  def concat(self, other):
    """Concatenates two dicts without copying internal data."""
    return TensorDict(self, **other)

  def copy(self):
    return TensorDict(super(TensorDict, self).copy())

  def __deepcopy__(self, memodict={}):
    return TensorDict(copy.deepcopy(list(self), memodict))

  def __getattr__(self, name):
    if not hasattr(torch.Tensor, name):
      raise AttributeError(
          '\'TensorDict\' object has not attribute \'{}\''.format(name))

    def apply_attr(*args, **kwargs):
      return TensorDict({
          n: getattr(e, name)(*args, **kwargs) if hasattr(e, name) else e
          for n, e in self.items()
      })

    return apply_attr

  def attribute(self, attr: str, *args):
    return TensorDict({n: getattr(e, attr, *args) for n, e in self.items()})

  def apply(self, fn, *args, **kwargs):
    return TensorDict({n: fn(e, *args, **kwargs) for n, e in self.items()})

  @staticmethod
  def _iterable(a):
    return isinstance(a, (TensorDict, list))


class TensorList(list):
  """Container mainly used for lists of torch tensors. Extends lists with pytorch functionality."""

  def __init__(self, list_of_tensors=None):
    if list_of_tensors is None:
      list_of_tensors = list()
    super(TensorList, self).__init__(list_of_tensors)

  def __deepcopy__(self, memodict={}):
    return TensorList(copy.deepcopy(list(self), memodict))

  def __getitem__(self, item):
    if isinstance(item, int):
      return super(TensorList, self).__getitem__(item)
    elif isinstance(item, (tuple, list)):
      return TensorList([super(TensorList, self).__getitem__(i) for i in item])
    else:
      return TensorList(super(TensorList, self).__getitem__(item))

  def __add__(self, other):
    if TensorList._iterable(other):
      return TensorList([e1 + e2 for e1, e2 in zip(self, other)])
    return TensorList([e + other for e in self])

  def __radd__(self, other):
    if TensorList._iterable(other):
      return TensorList([e2 + e1 for e1, e2 in zip(self, other)])
    return TensorList([other + e for e in self])

  def __iadd__(self, other):
    if TensorList._iterable(other):
      for i, e2 in enumerate(other):
        self[i] += e2
    else:
      for i in range(len(self)):
        self[i] += other
    return self

  def __sub__(self, other):
    if TensorList._iterable(other):
      return TensorList([e1 - e2 for e1, e2 in zip(self, other)])
    return TensorList([e - other for e in self])

  def __rsub__(self, other):
    if TensorList._iterable(other):
      return TensorList([e2 - e1 for e1, e2 in zip(self, other)])
    return TensorList([other - e for e in self])

  def __isub__(self, other):
    if TensorList._iterable(other):
      for i, e2 in enumerate(other):
        self[i] -= e2
    else:
      for i in range(len(self)):
        self[i] -= other
    return self

  def __mul__(self, other):
    if TensorList._iterable(other):
      return TensorList([e1 * e2 for e1, e2 in zip(self, other)])
    return TensorList([e * other for e in self])

  def __rmul__(self, other):
    if TensorList._iterable(other):
      return TensorList([e2 * e1 for e1, e2 in zip(self, other)])
    return TensorList([other * e for e in self])

  def __imul__(self, other):
    if TensorList._iterable(other):
      for i, e2 in enumerate(other):
        self[i] *= e2
    else:
      for i in range(len(self)):
        self[i] *= other
    return self

  def __truediv__(self, other):
    if TensorList._iterable(other):
      return TensorList([e1 / e2 for e1, e2 in zip(self, other)])
    return TensorList([e / other for e in self])

  def __rtruediv__(self, other):
    if TensorList._iterable(other):
      return TensorList([e2 / e1 for e1, e2 in zip(self, other)])
    return TensorList([other / e for e in self])

  def __itruediv__(self, other):
    if TensorList._iterable(other):
      for i, e2 in enumerate(other):
        self[i] /= e2
    else:
      for i in range(len(self)):
        self[i] /= other
    return self

  def __matmul__(self, other):
    if TensorList._iterable(other):
      return TensorList([e1 @ e2 for e1, e2 in zip(self, other)])
    return TensorList([e @ other for e in self])

  def __rmatmul__(self, other):
    if TensorList._iterable(other):
      return TensorList([e2 @ e1 for e1, e2 in zip(self, other)])
    return TensorList([other @ e for e in self])

  def __imatmul__(self, other):
    if TensorList._iterable(other):
      for i, e2 in enumerate(other):
        self[i] @= e2
    else:
      for i in range(len(self)):
        self[i] @= other
    return self

  def __mod__(self, other):
    if TensorList._iterable(other):
      return TensorList([e1 % e2 for e1, e2 in zip(self, other)])
    return TensorList([e % other for e in self])

  def __rmod__(self, other):
    if TensorList._iterable(other):
      return TensorList([e2 % e1 for e1, e2 in zip(self, other)])
    return TensorList([other % e for e in self])

  def __pos__(self):
    return TensorList([+e for e in self])

  def __neg__(self):
    return TensorList([-e for e in self])

  def __le__(self, other):
    if TensorList._iterable(other):
      return TensorList([e1 <= e2 for e1, e2 in zip(self, other)])
    return TensorList([e <= other for e in self])

  def __ge__(self, other):
    if TensorList._iterable(other):
      return TensorList([e1 >= e2 for e1, e2 in zip(self, other)])
    return TensorList([e >= other for e in self])

  def concat(self, other):
    return TensorList(super(TensorList, self).__add__(other))

  def copy(self):
    return TensorList(super(TensorList, self).copy())

  def unroll(self):
    if not any(isinstance(t, TensorList) for t in self):
      return self

    new_list = TensorList()
    for t in self:
      if isinstance(t, TensorList):
        new_list.extend(t.unroll())
      else:
        new_list.append(t)
    return new_list

  def list(self):
    return list(self)

  def attribute(self, attr: str, *args):
    return TensorList([getattr(e, attr, *args) for e in self])

  def apply(self, fn):
    return TensorList([fn(e) for e in self])

  def __getattr__(self, name):
    if not hasattr(torch.Tensor, name):
      raise AttributeError(
          '\'TensorList\' object has not attribute \'{}\''.format(name))

    def apply_attr(*args, **kwargs):
      return TensorList([getattr(e, name)(*args, **kwargs) for e in self])

    return apply_attr

  @staticmethod
  def _iterable(a):
    return isinstance(a, (TensorList, list))


def tensor_operation(op):

  def islist(a):
    return isinstance(a, TensorList)

  @functools.wraps(op)
  def oplist(*args, **kwargs):
    if len(args) == 0:
      raise ValueError(
          'Must be at least one argument without keyword (i.e. operand).')

    if len(args) == 1:
      if islist(args[0]):
        return TensorList([op(a, **kwargs) for a in args[0]])
    else:
      # Multiple operands, assume max two
      if islist(args[0]) and islist(args[1]):
        return TensorList(
            [op(a, b, *args[2:], **kwargs) for a, b in zip(*args[:2])])
      if islist(args[0]):
        return TensorList([op(a, *args[1:], **kwargs) for a in args[0]])
      if islist(args[1]):
        return TensorList(
            [op(args[0], b, *args[2:], **kwargs) for b in args[1]])

    # None of the operands are lists
    return op(*args, **kwargs)

  return oplist


class NestedTensor(object):

  def __init__(self, tensors, mask: Optional[Tensor]):
    self.tensors = tensors
    self.mask = mask

  def to(self, device):
    # type: (Device) -> NestedTensor # noqa
    cast_tensor = self.tensors.to(device)
    mask = self.mask
    if mask is not None:
      assert mask is not None
      cast_mask = mask.to(device)
    else:
      cast_mask = None
    return NestedTensor(cast_tensor, cast_mask)

  def decompose(self):
    return self.tensors, self.mask

  def __repr__(self):
    return str(self.tensors)


def _max_by_axis(the_list):
  # type: (List[List[int]]) -> List[int]
  maxes = the_list[0]
  for sublist in the_list[1:]:
    for index, item in enumerate(sublist):
      maxes[index] = max(maxes[index], item)
  return maxes

def nested_tensor_fix_size(tensor_list: List[Tensor], max_size):
  batch_shape = [len(tensor_list)] + max_size
  b, c, h, w = batch_shape
  dtype = tensor_list[0].dtype
  device = tensor_list[0].device
  tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
  mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
  for img, pad_img, m in zip(tensor_list, tensor, mask):
    pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
    m[:img.shape[1], :img.shape[2]] = False
  return NestedTensor(tensor, mask)
  

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
  # TODO make this more general
  if tensor_list[0].ndim == 3:
    if torchvision._is_tracing():
      # nested_tensor_from_tensor_list() does not export well to ONNX
      # call _onnx_nested_tensor_from_tensor_list() instead
      return _onnx_nested_tensor_from_tensor_list(tensor_list)

    # TODO make it support different-sized images
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
      pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
      m[:img.shape[1], :img.shape[2]] = False
  elif tensor_list[0].ndim == 4:
    if torchvision._is_tracing():
      # nested_tensor_from_tensor_list() does not export well to ONNX
      # call _onnx_nested_tensor_from_tensor_list() instead
      return _onnx_nested_tensor_from_tensor_list(tensor_list)

    # TODO make it support different-sized images
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
    batch_shape = [len(tensor_list)] + max_size
    b, t, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, t, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
      pad_img[:img.shape[0], :img.shape[1], :img.shape[2], :img.shape[3]].copy_(img)
      m[:img.shape[1], :img.shape[2], :img.shape[3]] = False
  else:
    raise ValueError('not supported')
  return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(
    tensor_list: List[Tensor]) -> NestedTensor:
  max_size = []
  for i in range(tensor_list[0].dim()):
    max_size_i = torch.max(
        torch.stack([img.shape[i] for img in tensor_list
                    ]).to(torch.float32)).to(torch.int64)
    max_size.append(max_size_i)
  max_size = tuple(max_size)

  # work around for
  # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
  # m[: img.shape[1], :img.shape[2]] = False
  # which is not yet supported in onnx
  padded_imgs = []
  padded_masks = []
  for img in tensor_list:
    padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
    padded_img = torch.nn.functional.pad(
        img, (0, padding[2], 0, padding[1], 0, padding[0]))
    padded_imgs.append(padded_img)

    m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
    padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]),
                                          "constant", 1)
    padded_masks.append(padded_mask.to(torch.bool))

  tensor = torch.stack(padded_imgs)
  mask = torch.stack(padded_masks)

  return NestedTensor(tensor, mask=mask)
