from enum import Enum, auto
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import (PeftModel, PeftConfig, PrefixTuningConfig, 
                 LoraConfig, get_peft_model, TaskType)
from omegaconf import DictConfig
from typing import Optional, Dict, Type

class FineTuningType(Enum):
    """Supported Fine-Tuning methods."""
    PEFT = auto()
    FULL = auto()

class PEFTType(Enum):
    """Supported Parameter-Efficient Fine-Tuning methods."""
    PREFIX = auto()
    LORA = auto()

class PEFTConfigFactory:
    """Factory for creating PEFT configurations using static mapping"""
    
    # Static mapping of PEFT types to their config classes
    PEFT_CONFIG_MAPPING: Dict[PEFTType, Type[PeftConfig]] = {
        PEFTType.LORA: (LoraConfig, 'lora'),
        PEFTType.PREFIX: (PrefixTuningConfig, 'prefix')
    }

    @classmethod
    def create_config(cls, peft_type: PEFTType, config: DictConfig) -> PeftConfig:
        try:
            config_class= cls.PEFT_CONFIG_MAPPING[peft_type]
            config_key = peft_type.name.lower()
            return config_class(**getattr(config, config_key))
        except KeyError:
            raise ValueError(f"Unsupported PEFT type: {peft_type}")
        except AttributeError:
            raise ValueError(f"Configuration for {config_key} not found in config")

class FineTuningStrategy(ABC):
    @abstractmethod
    def prepare_model(self, model: nn.Module, config: DictConfig) -> nn.Module:
        pass

class FullFineTuningStrategy(FineTuningStrategy):
    def prepare_model(self, model: nn.Module, config: DictConfig) -> nn.Module:
        num_trainable_layers = config.layers
        train_embeddings = config.train_embeddings
        
        if num_trainable_layers > len(model.encoder.layer):
            print(f"Warning: Number of trainable layers exceeds model layers. \
                   \nSetting it to {len(model.encoder.layer)}.")
            model.requires_grad_(True)
        elif num_trainable_layers == 0:
            model.requires_grad_(False)
        elif num_trainable_layers == -1:
            model.requires_grad_(True)
        else:
            self._set_layer_gradients(model, num_trainable_layers)
        
        model.embeddings.requires_grad_(train_embeddings)
        return model
    
    def _set_layer_gradients(self, model: nn.Module, num_trainable_layers: int):
        for layer in model.encoder.layer[-num_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in model.decoder.layer[:-num_trainable_layers]:
            for param in layer.parameters():
                param.requires_grad = False

class PEFTFineTuningStrategy(FineTuningStrategy):
    def __init__(self, peft_type: PEFTType):
        self.peft_type = peft_type

    def prepare_model(self, model: nn.Module, config: DictConfig) -> nn.Module:
        peft_config = PEFTConfigFactory.create_config(self.peft_type, config)
        return get_peft_model(model=model, peft_config=peft_config)


class ModelFineTuningManager:
    def __init__(self):
        self.strategies = {
            FineTuningType.FULL: FullFineTuningStrategy(),
            FineTuningType.PEFT: {
                PEFTType.LORA: PEFTFineTuningStrategy(PEFTType.LORA),
                PEFTType.PREFIX: PEFTFineTuningStrategy(PEFTType.PREFIX)
            }
        }

    def prepare_model(self, 
                     model: nn.Module, 
                     fine_tuning_type: FineTuningType,
                     peft_type: Optional[PEFTType],
                     config: DictConfig) -> nn.Module:
        if fine_tuning_type == FineTuningType.PEFT:
            if peft_type is None:
                raise ValueError("PEFT type must be specified for PEFT fine-tuning")
            strategy = self.strategies[fine_tuning_type][peft_type]
        else:
            strategy = self.strategies[fine_tuning_type]
        
        return strategy.prepare_model(model, config)


class BaseEncoder(nn.Module):
    def __init__(self,
                 model_id:str):
        super(BaseEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_id,
                                                 trust_remote_code=True)
        
    def forward(self, input_dict:dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**input_dict)
        return outputs.last_hidden_state

# class ModelFactory():
#     """Factory class for creating models"""
#     MODELS = {}
#     @classmethod
#     def create_model(self, config: DictConfig) -> nn.Module:
#         model_name = config.model_name
#         if model_name not in self.MODELS:
#             self.MODELS[model_name] = AutoModel.from_pretrained(model_name)
#         return self.MODELS[model_name]