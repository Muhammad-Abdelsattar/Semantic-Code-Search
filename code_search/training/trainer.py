import os
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from code_search.modeling.coordinator import ModelingCoordinator
from code_search.data.data_manager import DataManager

class TrainerManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = ModelingCoordinator(config=self.config.modeling)
        self.data_module = DataManager(config=self.config.data)
        self.trainer = self._build_trainer()

    def _build_trainer(self) -> L.Trainer:
        wandb_logger = WandbLogger(project="code-search",
                                  name="test-run")

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="./checkpoints",
            filename="best-model",
            save_top_k=1,
            mode="min",
        )

        trainer = L.Trainer(
            max_epochs=self.config.trainer.max_epochs,
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            strategy=self.config.trainer.strategy,
            precision=self.config.trainer.precision,
            gradient_clip_val=self.config.trainer.gradient_clip_val,
            accumulate_grad_batches=self.config.trainer.accumulate_grad_batches,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
        )
        return trainer
    
    def train(self):
        self.trainer.fit(self.model, datamodule=self.data_module)
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    config = OmegaConf.load("conf/config.yaml")
    trainer_manager = TrainerManager(config)
    trainer_manager.train()
