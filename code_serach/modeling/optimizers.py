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
                         config: DictConfig) -> torch.optim.Optimizer:
        """Creates an optimizer based on the given configuration."""
        optimizer_name = config.name.lower()
        optimizer_class = cls.OPTIMIZER_MAPPING.get(optimizer_name)
        if not optimizer_class:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        params = cls._get_parameter_groups(model, config.parameter_groups)
        
        optimizer = optimizer_class(params=params,
                                    lr=config.learning_rate,
                                    **config.optimizer_args)
        return optimizer

    @classmethod
    def create_scheduler(cls,
                         optimizer: torch.optim.Optimizer,
                         config: DictConfig) -> torch.optim.lr_scheduler._LRScheduler:
        """Creates a learning rate scheduler based on the given configuration."""
        scheduler_name = config.name.lower()
        scheduler_class = cls.SCHEDULER_MAPPING.get(scheduler_name)
        if not scheduler_class:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        scheduler = scheduler_class(optimizer=optimizer,
                                    **config.scheduler_args)
        return scheduler
    
    @classmethod
    def _get_parameter_groups(cls,
                              model: nn.Module,
                              parameter_groups_config: Optional[List[Dict]]
                              ) -> List[Dict]:
        """Creates parameter groups for the optimizer."""
        if not parameter_groups_config:
            return model.parameters()
        
        params = []
        for group_config in parameter_groups_config:
            group_params = []
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in group_config.layer_names):
                    group_params.append(param)
            params.append({"params": group_params, **group_config.optimizer_args})
        return params
