import os
from omegaconf import DictConfig
import lightning as L
from .utils import callbacks, logger

class TrainingManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.trainer = self._build_trainer()

    def _build_trainer(self) -> L.Trainer:

        trainer = L.Trainer(
            logger=logger,
            callbacks=callbacks,
            **self.config
        )
        return trainer
    
    def train(self,
              modeling_coordinator: L.LightningModule,
              datamodule: L.LightningDataModule,
              ckpt_path: str = None):

        self.trainer.fit(modeling_coordinator,
                            datamodule=datamodule,
                            ckpt_path=ckpt_path)