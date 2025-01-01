import torch
import torch.nn as nn
from lightning import LightningModule
from transformers import AutoModel
from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer
from .model_manager import ModelManager, FineTuningType, PEFTType
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
        loss_config.loss_args.embeddings_dim = self.embeddings_dim
        return LossFactory.create_loss(loss_config)

    def _build_model(self) -> nn.Module:
        model = AutoModel.from_pretrained(self.config.model.model_id, trust_remote_code=True)
        self.embeddings_dim = model.config.hidden_size #Usefule for other modules.
        model_manager = ModelManager()
        model = model_manager.prepare_model(model=model,
                                            config=self.config.model,)
        return model

    def configure_optimizers(self):
        optimizer = OptimizerFactory.create_optimizer(model=self.model,
                                                      config=self.config.optimizer)
        scheduler = OptimizerFactory.create_scheduler(optimizer=optimizer,
                                                      config=self.config.optimizer)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _common_step(self,
                     batch,
                     batch_idx,
                     mode: str):
        query = batch["query"]
        code = batch["code"]
        query_outputs = self.model(**query).last_hidden_state
        code_outputs = self.model(**code).last_hidden_state
        query_embeddings = torch.mean(query_outputs, dim=1)
        code_embeddings = torch.mean(code_outputs, dim=1)
        loss = self.loss_fn(query_embeddings, code_embeddings)
        self.log(f"{mode}_loss", loss,prog_bar=True,on_step=True,logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
