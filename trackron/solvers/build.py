# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import torch
import inspect
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from trackron.config import CfgNode
import torch.optim as optim
from timm.optim import AdamP, AdamW, Adafactor, Adahessian, Lookahead, Nadam, NovoGrad, NvNovoGrad, RAdam, RMSpropTF, SGDP
from timm.scheduler import CosineLRScheduler, PlateauLRScheduler, StepLRScheduler, TanhLRScheduler
# try:
#   from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
#   has_apex = True
# except ImportError:
#   has_apex = False

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]

_OPTIMIZER_DICT = {
    "sgd": optim.SGD,
    "momentum": optim.SGD,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "nadam": Nadam,
    "adamp": AdamP,
    "radam": RAdam,
    "sgdp": SGDP,
    "adadelta": optim.Adadelta,
    "adahessian": Adahessian,
    "rmsprop": optim.RMSprop,
    "rmsproptf": RMSpropTF,
    "novograd": NovoGrad,
    "nvnovograd": NvNovoGrad,
    # "fusedsgd": FusedSGD,
    # "fusedmomentum": FusedSGD,
    # "fusedadam": FusedAdam,
    # "fusedadamw": FusedAdam,
    # "fusedlamb": FusedLAMB,
    # "fusednovograd": FusedNovoGrad
}


class GradientClipType(Enum):
  VALUE = "value"
  NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
  """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
  cfg = copy.deepcopy(cfg)

  def clip_grad_norm(p: _GradientClipperInput):
    torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

  def clip_grad_value(p: _GradientClipperInput):
    torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

  _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
      GradientClipType.VALUE: clip_grad_value,
      GradientClipType.NORM: clip_grad_norm,
  }
  return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
  """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
  assert (per_param_clipper is None or global_clipper is None
         ), "Not allowed to use both per-parameter clipping and global clipping"

  def optimizer_wgc_step(self, closure=None):
    if per_param_clipper is not None:
      for group in self.param_groups:
        for p in group["params"]:
          per_param_clipper(p)
    else:
      # global clipper for future use with detr
      # (https://github.com/facebookresearch/detr/pull/287)
      all_params = itertools.chain(*[g["params"] for g in self.param_groups])
      global_clipper(all_params)
    super(type(self), self).step(closure)

  OptimizerWithGradientClip = type(
      optimizer.__name__ + "WithGradientClip",
      (optimizer,),
      {"step": optimizer_wgc_step},
  )
  return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
    cfg: CfgNode,
    optimizer: Type[torch.optim.Optimizer]) -> Type[torch.optim.Optimizer]:
  """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
  if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
    return optimizer
  if isinstance(optimizer, torch.optim.Optimizer):
    optimizer_type = type(optimizer)
  else:
    assert issubclass(optimizer, torch.optim.Optimizer), optimizer
    optimizer_type = optimizer

  grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
  OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
      optimizer_type, per_param_clipper=grad_clipper)
  if isinstance(optimizer, torch.optim.Optimizer):
    optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
    return optimizer
  else:
    return OptimizerWithGradientClip


def build_optimizer(cfg: CfgNode,
                    model: torch.nn.Module) -> torch.optim.Optimizer:
  """
    Build an optimizer from config.
    """
  # params = get_default_optimizer_params(
  #     model,
  #     base_lr=cfg.SOLVER.BASE_LR,
  #     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
  #     weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
  #     bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
  #     weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
  # )
  params = get_optimizer_params(cfg, model)
  optimizer_class = _OPTIMIZER_DICT.get(cfg.SOLVER.OPTIMIZER_NAME, optim.SGD)
  key_param = dict(inspect.signature(optimizer_class.__init__).parameters)
  kwargs = {
      "params": params,
      "lr": cfg.SOLVER.BASE_LR,
      "momentum": cfg.SOLVER.MOMENTUM,
      "nesterov": cfg.SOLVER.NESTEROV,
      "betas": cfg.SOLVER.BETAS,
      "eps": cfg.SOLVER.EPS,
      "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
  use_kwargs = {}
  for key in kwargs:
    if key in key_param:
      use_kwargs[key] = kwargs[key]
  return maybe_add_gradient_clipping(cfg, optimizer_class)(**use_kwargs)
  # return maybe_add_gradient_clipping(cfg, optimizer_class)(
  #     params,
  #     lr=cfg.SOLVER.BASE_LR,
  #     momentum=cfg.SOLVER.MOMENTUM,
  #     nesterov=cfg.SOLVER.NESTEROV,
  #     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
  # )


def get_optimizer_params(cfg: CfgNode,
                    model: torch.nn.Module) -> torch.optim.Optimizer:
  """
    Build an optimizer from config.
    """
  params: List[Dict[str, Any]] = []
  for key, value in model.named_parameters():
    if not value.requires_grad:
      continue

    lr = cfg.SOLVER.BASE_LR
    if key.startswith("backbone"):
      lr = lr * cfg.SOLVER.BASE_LR_BACKBONE
    weight_decay = cfg.SOLVER.WEIGHT_DECAY

    if key.endswith("norm.weight") or key.endswith("norm.bias"):
      weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
    elif key.endswith(".bias"):
      lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
      weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

  return params


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
):
  """
    Get default param list for optimizer, with support for a few types of
    overrides. If not overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
  if overrides is None:
    overrides = {}
  defaults = {}
  if base_lr is not None:
    defaults["lr"] = base_lr
  if weight_decay is not None:
    defaults["weight_decay"] = weight_decay
  bias_overrides = {}
  if bias_lr_factor is not None and bias_lr_factor != 1.0:
    if base_lr is None:
      raise ValueError("bias_lr_factor requires base_lr")
    bias_overrides["lr"] = base_lr * bias_lr_factor
  if weight_decay_bias is not None:
    bias_overrides["weight_decay"] = weight_decay_bias
  if len(bias_overrides):
    if "bias" in overrides:
      raise ValueError("Conflicting overrides for 'bias'")
    overrides["bias"] = bias_overrides

  norm_module_types = (
      torch.nn.BatchNorm1d,
      torch.nn.BatchNorm2d,
      torch.nn.BatchNorm3d,
      torch.nn.SyncBatchNorm,
      # NaiveSyncBatchNorm inherits from BatchNorm2d
      torch.nn.GroupNorm,
      torch.nn.InstanceNorm1d,
      torch.nn.InstanceNorm2d,
      torch.nn.InstanceNorm3d,
      torch.nn.LayerNorm,
      torch.nn.LocalResponseNorm,
  )
  params: List[Dict[str, Any]] = []
  memo: Set[torch.nn.parameter.Parameter] = set()
  for module in model.modules():
    for module_param_name, value in module.named_parameters(recurse=False):
      if not value.requires_grad:
        continue
      # Avoid duplicating parameters
      if value in memo:
        continue
      memo.add(value)

      hyperparams = copy.copy(defaults)
      if isinstance(module,
                    norm_module_types) and weight_decay_norm is not None:
        hyperparams["weight_decay"] = weight_decay_norm
      hyperparams.update(overrides.get(module_param_name, {}))
      params.append({"params": [value], **hyperparams})
  return params



def build_lr_scheduler(
    cfg: CfgNode,
    optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
  """
    Build a LR scheduler from config.
    """
  scheduler_name = cfg.SOLVER.LR_SCHEDULER.NAME
  num_iters = cfg.SOLVER.MAX_ITER
  lr_noise = cfg.SOLVER.LR_SCHEDULER.LR_NOISE

  if lr_noise is not None:
    if isinstance(lr_noise, (list, tuple)):
      noise_range = [n * num_iters for n in lr_noise]
      if len(noise_range) == 1:
        noise_range = noise_range[0]
    else:
      noise_range = lr_noise * num_iters
  else:
    noise_range = None

  lr_scheduler = None
  if scheduler_name == 'cosine':
    num_iters = num_iters - cfg.SOLVER.LR_SCHEDULER.COOL_DOWN
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_iters,
        t_mul=cfg.SOLVER.LR_SCHEDULER.LR_CYCLE_MUL,
        lr_min=cfg.SOLVER.LR_SCHEDULER.LR_MIN,
        decay_rate=cfg.SOLVER.LR_SCHEDULER.DECAY_RATE,
        warmup_lr_init=cfg.SOLVER.LR_SCHEDULER.WARMUP_LR,
        warmup_t=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
        cycle_limit=cfg.SOLVER.LR_SCHEDULER.LR_CYCLE_LIMIT,
        t_in_epochs=False,
        noise_range_t=noise_range,
        noise_pct=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_PCT,
        noise_std=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_STD,
        noise_seed=cfg.SEED,
    )
  elif scheduler_name == 'tanh':
    num_iters = num_iters - cfg.SOLVER.LR_SCHEDULER.COOL_DOWN
    lr_scheduler = TanhLRScheduler(
        optimizer,
        t_initial=num_iters,
        t_mul=cfg.SOLVER.LR_SCHEDULER.LR_CYCLE_MUL,
        lr_min=cfg.SOLVER.LR_SCHEDULER.LR_MIN,
        warmup_lr_init=cfg.SOLVER.LR_SCHEDULER.WARMUP_LR,
        warmup_t=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
        cycle_limit=cfg.SOLVER.LR_SCHEDULER.LR_CYCLE_LIMIT,
        t_in_epochs=False,
        noise_range_t=noise_range,
        noise_pct=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_PCT,
        noise_std=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_STD,
        noise_seed=cfg.SEED,
    )
  elif scheduler_name == 'step':
    lr_scheduler = StepLRScheduler(
        optimizer,
        decay_t=cfg.SOLVER.LR_SCHEDULER.DECAY_STEP,
        decay_rate=cfg.SOLVER.LR_SCHEDULER.DECAY_RATE,
        warmup_lr_init=cfg.SOLVER.LR_SCHEDULER.WARMUP_LR,
        warmup_t=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
        t_in_epochs=False,
        noise_range_t=noise_range,
        noise_pct=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_PCT,
        noise_std=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_STD,
        noise_seed=cfg.SEED,
    )
  elif scheduler_name == 'plateau':
    lr_scheduler = PlateauLRScheduler(
        optimizer,
        decay_rate=cfg.SOLVER.LR_SCHEDULER.DECAY_RATE,
        patience_t=cfg.SOLVER.LR_SCHEDULER.LR_PATIENCE_ITERS,
        lr_min=cfg.SOLVER.LR_SCHEDULER.LR_MIN,
        mode=cfg.SOLVER.LR_SCHEDULER.SCHEDULER_MODE,
        t_in_epochs=False,
        cooldown_t=0,
        warmup_lr_init=cfg.SOLVER.LR_SCHEDULER.WARMUP_LR,
        warmup_t=cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS,
        noise_range_t=noise_range,
        noise_pct=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_PCT,
        noise_std=cfg.SOLVER.LR_SCHEDULER.LR_NOISE_STD,
        noise_seed=cfg.SEED,
    )

  return lr_scheduler
