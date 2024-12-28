from typing import Optional
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig

from code_search.modeling.model_manager import ModelManager, Model, FineTuningType, PEFTType
from code_search.modeling.optimizers import OptimizerFactory
from code_search.modeling.losses import InfoNCELoss, MemoryBank

class ModelingCoordinator(LightningModule):
    def __init__(self,
                 config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        self.model_manager = ModelManager()
        self.model = self._build_model()
        self.loss_fn = self._build_loss_fn()
        self.memory_bank = self._build_memory_bank()

    def _build_loss_fn(self) -> nn.Module:
        return LossFactory.create_loss(self.config.loss)

    def _build_model(self) -> nn.Module:
        model = Model(self.config.model.model_id)
        fine_tuning_type = FineTuningType(self.config.model.fine_tuning_type.upper())
        peft_type = self.config.model.peft_type
        if peft_type:
            peft_type = PEFTType(peft_type.upper())
        model = self.model_manager.prepare_model(model=model,
                                                 config=self.config.model,
                                                 fine_tuning_type=fine_tuning_type,
                                                 peft_type=peft_type)
        return model

    def _build_loss_fn(self) -> nn.Module:
        return InfoNCELoss(**self.config.loss)

    def _build_memory_bank(self) -> Optional[MemoryBank]:
        if self.config.memory_bank.use_memory_bank:
            return MemoryBank(**self.config.memory_bank)
        return None

    def configure_optimizers(self):
        optimizer = OptimizerFactory.create_optimizer(model=self.model,
                                                      config=self.config.optimizer)
        scheduler = OptimizerFactory.create_scheduler(optimizer=optimizer,
                                                      config=self.config.optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Training step needs to be implemented")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Validation step needs to be implemented")

    def test_step(self, batch, batch_idx):
         raise NotImplementedError("Test step needs to be implemented")
