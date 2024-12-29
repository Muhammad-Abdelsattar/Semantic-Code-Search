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
        self.max_length = 512 # Default value, can be changed later

    def tokenize(self, queries: List[str], codes: List[str], max_length=None, padding=True, truncation=True):
        """Tokenizes the queries and codes.

        Args:
            queries: A list of query strings.
            codes: A list of code strings.
            max_length: The maximum length of the tokenized sequences.
            padding: Whether to pad the sequences to the maximum length.
            truncation: Whether to truncate the sequences to the maximum length.

        Returns:
            A dictionary containing the tokenized inputs.
        """
        tokenized_inputs = self.tokenizer(
            queries,
            codes,
            if max_length is None:
                max_length = self.max_length
            tokenized_inputs = self.tokenizer(
                queries,
                codes,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors="pt"
            )
            return tokenized_inputs
