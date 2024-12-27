import pytest
import os
import sys
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from code_search.modeling.optimizers import OptimizerFactory
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ExponentialLR, ConstantLR, SequentialLR


@pytest.fixture
def model():
    return nn.Linear(10, 10)

@pytest.fixture
def base_config():
    return OmegaConf.create({
        "optimizer": {
            "optimizer": {
                "name": "AdamW",
                "optimizer_args": {
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.999]
                }
            },
            "scheduler": {
                "name": "linear",
                "scheduler_args": {
                    "start_factor": 1/3,
                    "end_factor": 1.0,
                    "total_iters": 1000
                }
            },
            "warmup": {
                "warmup_steps": 100,
                "start_factor": 1/3,
                "end_factor": 1.0
            }
        }
    })

def test_create_optimizer(model, base_config):
    # Test creating AdamW optimizer
    optimizer = OptimizerFactory.create_optimizer(model, base_config, learning_rate=1e-3)
    assert isinstance(optimizer, AdamW)
    
    # Test creating Adam optimizer
    base_config.optimizer.optimizer.name = "Adam"
    optimizer = OptimizerFactory.create_optimizer(model, base_config, learning_rate=1e-3)
    assert isinstance(optimizer, Adam)

    # Test creating SGD optimizer
    base_config.optimizer.optimizer.name = "SGD"
    optimizer = OptimizerFactory.create_optimizer(model, base_config, learning_rate=1e-3)
    assert isinstance(optimizer, SGD)

    # Test invalid optimizer name
    base_config.optimizer.optimizer.name = "invalid_optimizer"
    with pytest.raises(ValueError, match="Unsupported optimizer: invalid_optimizer"):
        OptimizerFactory.create_optimizer(model, base_config, learning_rate=1e-3)

def test_create_scheduler(model, base_config):
    optimizer = OptimizerFactory.create_optimizer(model, base_config, learning_rate=1e-3)

    # Test creating linear scheduler with warmup
    scheduler = OptimizerFactory.create_scheduler(optimizer, base_config)
    assert isinstance(scheduler, SequentialLR)
    assert isinstance(scheduler.scheduler1, LinearLR)
    assert isinstance(scheduler.scheduler2, LinearLR)

    # Test creating linear scheduler without warmup
    base_config.optimizer.warmup = None
    scheduler = OptimizerFactory.create_scheduler(optimizer, base_config)
    assert isinstance(scheduler, LinearLR)

    # Test creating cosine scheduler
    base_config.optimizer.warmup = {
                "warmup_steps": 100,
                "start_factor": 1/3,
            }
    base_config.optimizer.scheduler.name = "cosine"
    base_config.optimizer.scheduler.scheduler_args.T_max = 100
    scheduler = OptimizerFactory.create_scheduler(optimizer, base_config)
    assert isinstance(scheduler, SequentialLR)
    assert isinstance(scheduler.scheduler1, LinearLR)
    assert isinstance(scheduler.scheduler2, CosineAnnealingLR)

    # Test invalid scheduler name
    base_config.optimizer.scheduler.name = "invalid_scheduler"
    with pytest.raises(ValueError, match="Unsupported scheduler: invalid_scheduler"):
         OptimizerFactory.create_scheduler(optimizer, base_config)
