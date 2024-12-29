from typing import List
import json
import pandas as pd
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoTokenizer
from torch.utils.data import Dataset

class CodeSearchDataset(Dataset):
    def __init__(self,
                 config: DictConfig,):
        """
        Initializes the dataset.

        Args:
            file_paths: A list of paths to JSONL files.
            tokenizer: A CodeSearchTokenizer instance.
        """
        self.config = config
        self.data = self._load_data_from_jsonl(self.config.file_paths)

    def _load_data_from_jsonl(self, file_paths: List[str]) -> pd.DataFrame:
        """Loads data from one or more JSONL files into a pandas DataFrame."""
        all_data = []
        for file_path in file_paths:
            try:
                data = pd.read_json(file_path, lines=True)
                all_data.append(data)
            except ValueError:
                print(f"Warning: File wans't read: {file_path}")
                continue
        return pd.concat(all_data, ignore_index=True)

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
