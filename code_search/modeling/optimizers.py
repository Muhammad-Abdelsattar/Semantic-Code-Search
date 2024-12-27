import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ExponentialLR, ConstantLR, SequentialLR
from omegaconf import DictConfig
from typing import List, Dict, Tuple, Optional, Callable


class OptimizerFactory:
    """Factory class for creating optimizers and learning rate schedulers."""

    OPTIMIZER_MAPPING: Dict[str, Callable] = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
    }

    SCHEDULER_MAPPING: Dict[str, Callable] = {
        "linear": LinearLR,
        "cosine": CosineAnnealingLR,
    }

    @classmethod
    def create_optimizer(cls,
                         model: nn.Module,
                         config: DictConfig,
                         learning_rate: float) -> torch.optim.Optimizer:
        """Creates an optimizer based on the given configuration."""
        optimizer_config = config.optimizer.optimizer
        optimizer_name = optimizer_config.name.lower()
        optimizer_class = cls.OPTIMIZER_MAPPING.get(optimizer_name)
        if not optimizer_class:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Filter out invalid arguments for the optimizer
        valid_args = {}
        if optimizer_name in ["adam", "adamw"]:
            valid_args = optimizer_config.optimizer_args
        else:
            valid_args = {k: v for k, v in optimizer_config.optimizer_args.items() if k not in ["betas"]}
        
        optimizer = optimizer_class(params=model.parameters(),
                                    lr=learning_rate,
                                    **valid_args)
        return optimizer

    @classmethod
    def create_scheduler(cls,
                         optimizer: torch.optim.Optimizer,
                         config: DictConfig) -> torch.optim.lr_scheduler._LRScheduler:
        """Creates a learning rate scheduler based on the given configuration."""
        scheduler_config = config.optimizer.scheduler
        scheduler_name = scheduler_config.name.lower()
        scheduler_class = cls.SCHEDULER_MAPPING.get(scheduler_name)
        if not scheduler_class:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        scheduler_args = {k: v for k, v in scheduler_config.scheduler_args.items() if k not in ["start_factor", "end_factor", "total_iters", "total_epochs"]}
        main_scheduler = scheduler_class(optimizer=optimizer,
                                    **scheduler_args)
        
        if "warmup" in config.optimizer and config.optimizer.warmup:
            warmup_config = config.optimizer.warmup
            warmup_scheduler = LinearLR(optimizer=optimizer,
                                          start_factor=warmup_config.start_factor,
                                          end_factor=1.0,
                                          total_iters=warmup_config.warmup_steps)
            scheduler = SequentialLR(optimizer=optimizer,
                                     schedulers=[warmup_scheduler, main_scheduler],
                                     milestones=[warmup_config.warmup_steps])
            return scheduler
        else:
            return main_scheduler
