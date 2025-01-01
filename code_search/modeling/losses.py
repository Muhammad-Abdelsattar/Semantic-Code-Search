from typing import Optional, Dict
from typing import Dict, Type
from omegaconf import OmegaConf, DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingsBank:
    def __init__(self,
                 embeddings_dim: int,
                 bank_size: int):
        self.embeddings_dim = embeddings_dim
        self.bank_size = bank_size
        self.bank = torch.zeros(self.bank_size, self.embeddings_dim)
        self.ptr = 0

    def update(self, embeddings: torch.Tensor):
        batch_size = embeddings.size(0)
        if (self.ptr + batch_size) > self.bank_size:
            self.ptr = 0
        # if self.bank_size %  batch_size != 0:
        #     raise ValueError("Batch size must be a multiple of the memory bank size.")
        self.bank[self.ptr:self.ptr + batch_size] = embeddings
        self.ptr = (self.ptr + batch_size) % self.bank_size

class InfoNCELoss(nn.Module):
    def __init__(self,
                 temperature: float = 0.07,
                 bank_size: Optional[int] = None,
                 embeddings_dim: Optional[int] = None,):
        super().__init__()
        self.temperature = temperature
        if bank_size is not None and embeddings_dim is not None:
            self.bank = EmbeddingsBank(embeddings_dim, bank_size)
        else:
            self.bank = None

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

        if self.bank is not None:
            all_key_embeddings = torch.cat([key_embeddings, 
                                            self.bank.bank.detach().to(device=key_embeddings.device,
                                                                       dtype=key_embeddings.dtype)],
                                           dim=0,
                                           dtype=key_embeddings.dtype)
        else:
            all_key_embeddings = key_embeddings
        # Compute similarity scores

        logits = torch.matmul(query_embeddings, all_key_embeddings.T) / self.temperature

        if self.bank is not None:
            self.bank.update(key_embeddings)

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
        if "bank_size" in loss_args:
            if loss_args.embeddings_dim is None:
                raise ValueError("Embeddings dimension must be provided for memory bank.")
        loss_class = cls.LOSS_MAPPING.get(loss_name)
        if not loss_class:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        return loss_class(**loss_args)
