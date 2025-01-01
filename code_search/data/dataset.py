from typing import List
import json
import pandas as pd
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
from torch.utils.data import Dataset

class CodeSearchDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,):

        self.data = data

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx: The index of the item to retrieve.

        Returns:
            A dictionary containing the tokenized inputs.
        """
        item = self.data.iloc[idx]
        query = item['query']
        code = item['code']
        return {
            "query": query,
            "code": code
        }
