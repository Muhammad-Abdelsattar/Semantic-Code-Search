from typing import Optional, Dict
from typing import Dict, Type
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank:
    def __init__(self, size: int, embedding_dim: int, device: torch.device):
        self.size = size
        self.embedding_dim = embedding_dim
        self.device = device
        self.bank = torch.zeros(size, embedding_dim, device=device)
        self.ptr = 0
        self.full = False

    def update(self, embeddings: torch.Tensor):
        batch_size = embeddings.size(0)
        if(self.size % batch_size != 0):
            raise ValueError("Batch size must be a multiple of memory bank size")
        self.bank[self.ptr:self.ptr + batch_size] = embeddings
        self.ptr = (self.ptr + batch_size) % self.size
        if self.ptr == 0:
            self.full = True

class InfoNCELoss(nn.Module):
    def __init__(self,
                 temperature: float = 0.07,
                 memory_bank_size: Optional[int] = None,
                 embedding_dim: Optional[int] = None):
        super().__init__()
        self.temperature = temperature
        self.memory_bank = None
        if memory_bank_size is not None and embedding_dim is not None:
            self.memory_bank = MemoryBank(size=memory_bank_size,
                                          embedding_dim=embedding_dim,
                                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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

        if self.memory_bank is not None:
            # Combine current key embeddings with memory bank
            all_key_embeddings = torch.cat([key_embeddings, self.memory_bank.bank.clone().detach()], dim=0)
        else:
            all_key_embeddings = key_embeddings
        
        if self.memory_bank is not None:
            self.memory_bank.update(key_embeddings)

        # Compute similarity scores
        logits = torch.matmul(query_embeddings, all_key_embeddings.T) / self.temperature

        # Create target labels
        batch_size = key_embeddings.size(0)
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
                    config: Dict) -> nn.Module:
        loss_name = config.name
        loss_args = config.get("loss_args", {})
        loss_class = cls.LOSS_MAPPING.get(loss_name)
        if not loss_class:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        return loss_class(**loss_args)
