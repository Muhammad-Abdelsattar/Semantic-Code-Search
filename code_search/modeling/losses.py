from typing import Optional, Dict
from typing import Dict, Type
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self,
                 temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                query_embeddings: torch.Tensor,
                key_embeddings: torch.Tensor,
                normalize: bool = True) -> torch.Tensor:
        """
        Computes the InfoNCE loss with optional memory bank.
        """
        if normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            key_embeddings = F.normalize(key_embeddings, p=2, dim=1)

        # Compute similarity scores

        logits = torch.matmul(query_embeddings, key_embeddings.T) / self.temperature

        # Create target labels
        batch_size = query_embeddings.size(0)
        labels = torch.arange(batch_size, device=query_embeddings.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

class LossFactory:
    """Factory class for creating loss functions."""
    LOSS_MAPPING: Dict[str, Type[nn.Module]] = {
        "InfoNCE": InfoNCELoss
    }

    @classmethod
    def create_loss(cls,
                    config: DictConfig) -> nn.Module:
        loss_name = config.get("loss_name", "InfoNCE")
        loss_args = config.get("loss_args", {})
        loss_class = cls.LOSS_MAPPING.get(loss_name)
        if not loss_class:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        return loss_class(**loss_args)
