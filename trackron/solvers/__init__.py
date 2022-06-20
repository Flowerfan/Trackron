import torch
from .build import build_lr_scheduler, build_optimizer, get_default_optimizer_params
from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR, LRMultiplier, WarmupParamScheduler

__all__ = [k for k in globals().keys() if not k.startswith("_")]



def build_optimizer_scheduler(cfg, model):
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    return optimizer, lr_scheduler
    
