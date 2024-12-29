import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig
from .model_manager import ModelManager, Model, FineTuningType, PEFTType
from .optimizers import OptimizerFactory
from .losses import LossFactory

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
        loss_config["loss_args"]["embedding_dim"] = self.config.model.embedding_dim
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

    def _common_step(self,
                     batch,
                     batch_idx,
                     mode: str):
        query = batch["query"]
        code = batch["code"]
        query_outputs = self.model(**query)
        code_outputs = self.model(**code)
        query_embeddings = torch.mean(query_outputs, dim=1)
        code_embeddings = torch.mean(code_outputs, dim=1)
        loss = self.loss_fn(query_embeddings, code_embeddings)
        self.log(f"{mode}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
