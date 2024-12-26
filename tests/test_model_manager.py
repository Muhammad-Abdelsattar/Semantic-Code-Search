import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from code_serach.modeling.model_manager import (
    ModelManager,
    FineTuningType,
    PEFTType,
    Model
)

@pytest.fixture
def config():
    conf = """
    model:
      model_id: "jinaai/jina-embeddings-v2-base-code"
      fine_tuning:
        full:
          layers: 2
          train_embeddings: false
        peft:
          lora:
            task_type: "SEQ_CLS"
            inference_mode: False
            target_modules: ["query", "value", "key"]
            r: 8
            lora_alpha: 32
            lora_dropout: 0.05
          prefix:
            task_type: "SEQ_CLS"
            num_virtual_tokens: 20
            inference_mode: False
            prefix_projection: True
    """
    return OmegaConf.create(conf)

@pytest.fixture
def model(config):
    return Model(model_id=config.model.model_id)

@pytest.fixture
def model_manager():
    return ModelManager()

def test_full_fine_tuning(model_manager, model, config):
    
    prepared_model = model_manager.prepare_model(model=model,
                                                 config=config.model.fine_tuning,
                                                 fine_tuning_type=FineTuningType.FULL)
    
    assert isinstance(prepared_model, torch.nn.Module)
    
    # Check if the last 2 layers are trainable
    for name, param in prepared_model.named_parameters():
        if "encoder.layer.10" in name or "encoder.layer.11" in name:
            assert param.requires_grad == True
        elif "embeddings" in name:
            assert param.requires_grad == False
        else:
            assert param.requires_grad == False

def test_full_fine_tuning_all_layers(model_manager, model, config):
    config.model.fine_tuning.full.layers = -1
    prepared_model = model_manager.prepare_model(model=model,
                                                 config=config.model.fine_tuning,
                                                 fine_tuning_type=FineTuningType.FULL)
    
    assert isinstance(prepared_model, torch.nn.Module)
    
    # Check if all layers are trainable
    for name, param in prepared_model.named_parameters():
        if "embeddings" not in name:
            assert param.requires_grad == True
        else:
            assert param.requires_grad == False

def test_full_fine_tuning_no_layers(model_manager, model, config):
    config.model.fine_tuning.full.layers = 0
    prepared_model = model_manager.prepare_model(model=model,
                                                 config=config.model.fine_tuning,
                                                 fine_tuning_type=FineTuningType.FULL)
    
    assert isinstance(prepared_model, torch.nn.Module)
    
    # Check if no layers are trainable
    for name, param in prepared_model.named_parameters():
        assert param.requires_grad == False

def test_lora_fine_tuning(model_manager, model, config):
    prepared_model = model_manager.prepare_model(model=model,
                                                 config=config.model.fine_tuning,
                                                 fine_tuning_type=FineTuningType.PEFT,
                                                 peft_type=PEFTType.LORA)
    assert isinstance(prepared_model, torch.nn.Module)
    assert hasattr(prepared_model, 'base_model')
    assert hasattr(prepared_model.base_model, 'prefix_tokens')
    assert hasattr(prepared_model.base_model.model, 'lora_dropout')
    for name, module in prepared_model.base_model.model.named_modules():
        if "lora_A" in name:
            assert hasattr(module, 'lora_A')
        if "lora_B" in name:
            assert hasattr(module, 'lora_B')

def test_prefix_fine_tuning(model_manager, model, config):
    prepared_model = model_manager.prepare_model(model=model,
                                                 config=config.model.fine_tuning,
                                                 fine_tuning_type=FineTuningType.PEFT,
                                                 peft_type=PEFTType.PREFIX)
    assert isinstance(prepared_model, torch.nn.Module)
    assert hasattr(prepared_model, 'base_model')

def test_peft_fine_tuning_no_peft_type(model_manager, model, config):
    with pytest.raises(ValueError, match="PEFT type must be specified for PEFT fine-tuning"):
         model_manager.prepare_model(model=model,
                                     config=config.model.fine_tuning,
                                     fine_tuning_type=FineTuningType.PEFT)
    
def test_invalid_stage(model_manager, model, config):
    with pytest.raises(NotImplementedError, match="Inference stage is not implemented yet."):
        model_manager.prepare_model(model=model,
                                    config=config.model.fine_tuning,
                                    fine_tuning_type=FineTuningType.FULL,
                                    stage="inference")
