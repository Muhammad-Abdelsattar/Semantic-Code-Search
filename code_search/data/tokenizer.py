from transformers import AutoTokenizer
from typing import List
from omegaconf import DictConfig

class CodeSearchTokenizer:
    def __init__(self, config: DictConfig):
        """Initializes the tokenizer.

        Args:
            config: A DictConfig object containing the model configuration.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)

    def tokenize(self, queries: List[str], codes: List[str]):
        """Tokenizes the queries and codes.

        Args:
            queries: A list of query strings.
            codes: A list of code strings.

        Returns:
            A dictionary containing the tokenized inputs.
        """
        tokenized_inputs = self.tokenizer(
            queries,
            codes,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return tokenized_inputs
