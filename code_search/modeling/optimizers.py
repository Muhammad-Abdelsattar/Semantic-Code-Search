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

    WARMUP_MAPPING: Dict[str, Callable] = {
        "linear": LinearLR,
        "exponential": ExponentialLR,
        "constant": ConstantLR,
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

        warmup_config = config.optimizer.get("warmup")
        if warmup_config:
            warmup_name = warmup_config.name.lower()
            warmup_class = cls.WARMUP_MAPPING.get(warmup_name)
            if not warmup_class:
                raise ValueError(f"Unsupported warmup scheduler: {warmup_name}")
            
            warmup_steps = warmup_config.get("warmup_steps")
            warmup_epochs = warmup_config.get("warmup_epochs")
            if warmup_steps is not None and warmup_epochs is not None:
                raise ValueError("Specify either warmup_steps or warmup_epochs, not both.")
            if warmup_steps is None and warmup_epochs is None:
                raise ValueError("Specify either warmup_steps or warmup_epochs.")
            
            if warmup_steps:
                warmup_scheduler = warmup_class(optimizer=optimizer,
                                                start_factor=warmup_config.get("start_factor", 1/3),
                                                end_factor=warmup_config.get("end_factor", 1.0),
                                                total_iters=warmup_steps)
            else: #warmup_epochs
                 warmup_scheduler = warmup_class(optimizer=optimizer,
                                                start_factor=warmup_config.get("start_factor", 1/3),
                                                end_factor=warmup_config.get("end_factor", 1.0),
                                                total_epochs=warmup_epochs)
            
            scheduler_args = {k: v for k, v in scheduler_config.scheduler_args.items() if k not in ["start_factor", "end_factor", "total_iters", "total_epochs"]}
            scheduler = scheduler_class(optimizer=optimizer,
                                        **scheduler_args)
            
            scheduler = SequentialLR(optimizer=optimizer,
                                     schedulers=[warmup_scheduler, scheduler],
                                     milestones=[warmup_steps if warmup_steps else warmup_epochs])
            return scheduler
        else:
            scheduler_args = {k: v for k, v in scheduler_config.scheduler_args.items() if k not in ["start_factor", "end_factor", "total_iters", "total_epochs"]}
            scheduler = scheduler_class(optimizer=optimizer,
                                        **scheduler_args)
            return scheduler
