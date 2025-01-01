from lightning.pytorch.callbacks import (ModelCheckpoint,
                                         ModelSummary,
                                         TQDMProgressBar)
from lightning.pytorch.loggers import WandbLogger


callbacks = [
    ModelCheckpoint(monitor="val_loss",
                     dirpath=".artifacts/checkpoints",
                     filename="best-ckpt",
                     save_top_k=1,
                     mode="min"),
    ModelSummary(max_depth=1),
    TQDMProgressBar(refresh_rate=1),
]

logger = WandbLogger(project="code-search")