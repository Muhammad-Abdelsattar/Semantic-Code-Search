import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
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
        
        optimizer = optimizer_class(params=model.parameters(),
                                    lr=learning_rate,
                                    **optimizer_config.optimizer_args)
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
        
        scheduler = scheduler_class(optimizer=optimizer,
                                    **scheduler_config.scheduler_args)
        return scheduler
