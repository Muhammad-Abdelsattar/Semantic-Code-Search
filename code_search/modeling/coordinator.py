from typing import Optional
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig

from code_search.modeling.model_manager import ModelManager, Model, FineTuningType, PEFTType
from code_search.modeling.optimizers import OptimizerFactory
from code_search.modeling.losses import LossFactory, InfoNCELoss

class ModelingCoordinator(LightningModule):
    def __init__(self,
                 config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        self.model = self._build_model()
        self.loss_fn = self._build_loss_fn()

    def _build_loss_fn(self) -> nn.Module:
        loss_config = self.config.loss
        embedding_dim = self.config.model.embedding_dim
        loss_config["loss_args"]["embedding_dim"] = embedding_dim
        return LossFactory.create_loss(loss_config)

    def _build_model(self) -> nn.Module:
        model = Model(self.config.model.model_id)
        fine_tuning_type = FineTuningType(self.config.model.fine_tuning_type.upper())
        model_manager = ModelManager()
        peft_type = self.config.model.peft_type
        if peft_type:
            peft_type = PEFTType(peft_type.upper())
        model = model_manager.prepare_model(model=model,
                                                 config=self.config.model,
                                                 fine_tuning_type=fine_tuning_type,
                                                 peft_type=peft_type)
        return model

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
