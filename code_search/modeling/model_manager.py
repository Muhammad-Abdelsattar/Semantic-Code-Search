from enum import Enum, auto
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import (PeftModel, PeftConfig, PrefixTuningConfig, 
                 LoraConfig, get_peft_model, TaskType)
from omegaconf import DictConfig
from typing import Optional, Dict, Type

class FineTuningType(str,Enum):
    """Supported Fine-Tuning methods."""
    PEFT = "PEFT"
    FULL = "FULL"

class PEFTType(str,Enum):
    """Supported Parameter-Efficient Fine-Tuning methods."""
    PREFIX = "PREFIX"
    LORA = "LORA"

class PEFTConfigFactory:
    """Factory for creating PEFT configurations using static mapping"""
    
    # Static mapping of PEFT types to their config classes
    PEFT_CONFIG_MAPPING: Dict[PEFTType, Type[PeftConfig]] = {
        PEFTType.LORA: LoraConfig,
        PEFTType.PREFIX: PrefixTuningConfig
    }

    @classmethod
    def create_config(cls,
                      peft_type: PEFTType,
                      config: DictConfig) -> PeftConfig:
        config_class = cls.PEFT_CONFIG_MAPPING[peft_type]
        config_key = peft_type.name.lower()
        return config_class(**getattr(config, config_key))

class FineTuningStrategy(ABC):
    @abstractmethod
    def prepare_model(self,
                      model: nn.Module,
                      config: DictConfig) -> nn.Module:
        pass

class FullFineTuningStrategy(FineTuningStrategy):
    def prepare_model(self,
                      model: nn.Module,
                      config: DictConfig) -> nn.Module:
        num_trainable_layers = config.layers
        train_embeddings = config.train_embeddings
        model.requires_grad_(False)
        if num_trainable_layers == -1:
            model.requires_grad_(True)
        elif num_trainable_layers > 0:
            model.pooler.requires_grad_(True)
            for layer in model.encoder.layer[-num_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            model.pooler.requires_grad_(True)
        model.embeddings.requires_grad_(train_embeddings)
        return model
 
class PEFTFineTuningStrategy(FineTuningStrategy):
    def __init__(self,
                 peft_type: PEFTType):
        self.peft_type = peft_type

    def prepare_model(self,
                      model: nn.Module,
                      config: DictConfig) -> nn.Module:
        peft_config = PEFTConfigFactory.create_config(self.peft_type, config)
        return get_peft_model(model=model, peft_config=peft_config)


class ModelManager:
    def __init__(self):
        self.strategies = {
            (FineTuningType.FULL, None): FullFineTuningStrategy(),
            (FineTuningType.PEFT, PEFTType.LORA): PEFTFineTuningStrategy(PEFTType.LORA),
            (FineTuningType.PEFT, PEFTType.PREFIX): PEFTFineTuningStrategy(PEFTType.PREFIX)
        }

    def prepare_model(self,
                     model: nn.Module,
                     config: DictConfig,
                    #  fine_tuning_type: FineTuningType,
                    #  peft_type: Optional[PEFTType] = None,
                     stage: str = "fine_tuning") -> nn.Module:
        if stage == "fine_tuning":
            if(len(config.fine_tuning.use.split("/")) == 2):
                peft_type = config.fine_tuning.use.split("/")[1].upper()
            else:
                peft_type = None
            fine_tuning_type = config.fine_tuning.use.split("/")[0].upper()
            strategy = self.strategies.get((fine_tuning_type, peft_type))
            if not strategy:
                raise ValueError(f"Unsupported fine-tuning type: {fine_tuning_type} and peft type: {peft_type}")
            
            if fine_tuning_type == FineTuningType.PEFT:
                model = strategy.prepare_model(model, config.fine_tuning.peft)
            else:
                model = strategy.prepare_model(model, config.fine_tuning.full)
            return model
            # return torch.compile(model)

        elif stage == "inference":
            raise NotImplementedError("Inference stage is not implemented yet.")