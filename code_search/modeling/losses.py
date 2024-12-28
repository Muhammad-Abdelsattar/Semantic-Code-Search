from typing import Optional
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
                memory_bank: Optional['MemoryBank'] = None,
                normalize: bool = True) -> torch.Tensor:
        """
        Computes the InfoNCE loss with optional memory bank.
        """
        if normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            key_embeddings = F.normalize(key_embeddings, p=2, dim=1)

        if memory_bank is not None:
            # Combine current key embeddings with memory bank
            all_key_embeddings = torch.cat([key_embeddings, memory_bank.bank.clone().detach()], dim=0)
        else:
            all_key_embeddings = key_embeddings

        # Compute similarity scores
        logits = torch.matmul(query_embeddings, all_key_embeddings.T) / self.temperature

        # Create target labels
        batch_size = key_embeddings.size(0)
        labels = torch.arange(batch_size, device=query_embeddings.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

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
