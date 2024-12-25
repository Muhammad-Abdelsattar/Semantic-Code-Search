import torch
from omegaconf import DictConfig, OmegaConf
from code_serach.modeling.model_manager import (
    BaseEncoder,
    ModelFineTuningManager,
    FineTuningType,
    PEFTType
)

def create_lora_model(config: DictConfig) -> torch.nn.Module:
    """
    Creates a LORA model based on the provided configuration.

    Args:
        config (DictConfig): Configuration for the model and fine-tuning.

    Returns:
        torch.nn.Module: The prepared LORA model.
    """
    model_id = config.model.model_id
    base_model = BaseEncoder(model_id=model_id)
    
    fine_tuning_manager = ModelFineTuningManager()
    
    lora_model = fine_tuning_manager.prepare_model(
        model=base_model.encoder,
        fine_tuning_type=FineTuningType.PEFT,
        peft_type=PEFTType.LORA,
        config=config.model.fine_tuning
    )
    
    return lora_model

if __name__ == '__main__':
    # Load the configuration from the YAML file
    config = OmegaConf.load("conf/modeling.yaml")
    
    # Create the LORA model
    lora_model = create_lora_model(config)
    
    # Print the model to verify
    print(lora_model)
