import json
import pandas as pd
from typing import List
from torch.utils.data import Dataset
from code_search.data.tokenizer import CodeSearchTokenizer

class CodeSearchDataset(Dataset):
    def __init__(self, file_paths: List[str], tokenizer: CodeSearchTokenizer):
        """
        Initializes the dataset.

        Args:
            file_paths: A list of paths to JSONL files.
            tokenizer: A CodeSearchTokenizer instance.
        """
        self.data = self._load_data_from_jsonl(file_paths)
        self.tokenizer = tokenizer

    def _load_data_from_jsonl(self, file_paths: List[str]) -> pd.DataFrame:
        """Loads data from one or more JSONL files into a pandas DataFrame."""
        all_data = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        json_line = json.loads(line)
                        all_data.append(json_line)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line in {file_path}: {line}")
                        continue
        df = pd.DataFrame(all_data)
        if 'query' not in df.columns or 'code' not in df.columns:
            raise ValueError("JSONL files must contain 'query' and 'code' fields.")
        return df

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
        tokenized_inputs = self.tokenizer.tokenize([query], [code])
        return tokenized_inputs
