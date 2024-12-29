import os
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from code_search.modeling.coordinator import ModelingCoordinator
from code_search.data.data_manager import DataManager

def train(config: DictConfig):
    model = ModelingCoordinator(config=config.modeling)
    data_manager = DataManager(config=config.data)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join("checkpoints"),
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
    )
    
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
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        strategy=config.trainer.strategy,
        precision=config.trainer.precision,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    
    trainer.fit(model, datamodule=data_manager)
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    config = OmegaConf.load("conf/config.yaml")
    train(config)
