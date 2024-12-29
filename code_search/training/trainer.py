import os
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig
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
    
    logger = TensorBoardLogger(save_dir="logs")
    
    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="gpu",
    )
    
    trainer.fit(model, datamodule=data_manager)
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    config = OmegaConf.load("conf/config.yaml")
    train(config)
