from .base_trainer import DefaultTrainer
# from .ltr_trainer import LTRTrainer


def build_trainer(cfg, model, loaders, optimizer, lr_scheduler):
    # trainer = LTRTrainer(cfg, objective, loaders, optimizer, lr_scheduler=lr_scheduler)
    trainer = DefaultTrainer(cfg, model=model, loaders=loaders, optimizer=optimizer, lr_scheduler=lr_scheduler)
    return trainer

