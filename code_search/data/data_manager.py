from dataclasses import dataclass
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.data import DataCollatorForSeq2Seq
from .dataset import CodeSearchDataset


class DataCollator:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,):
        self.tokenizer = tokenizer
        
    def __call__(self,
                 input_data: list[dict[str,str]]):
        return self.collate(input_data=input_data)

    def collate(self,
                input_data: list[dict[str,str]]):
        data_dict = {}
        collated = {}
        for item in input_data:
            for k,v in item.items():
                if k not in data_dict:
                    data_dict[k] = []
                data_dict[k].append(v)
        for k,v in data_dict.items():
            collated[k] = self.tokenizer(v,return_tensors="pt",padding="longest")
        return collated
        
class DataManager(LightningDataModule):
    def __init__(self,
                 config: DictConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_id)
        self.collator = DataCollator(tokenizer=self.tokenizer)

    def setup(self, stage: str) -> None:
        data = self._load_data_from_jsonl(self.config.train_val_files)
        if stage == "fit":
            self.train_dataset = CodeSearchDataset(data[:int(len(data) * self.config.train_split)])
            self.val_dataset = CodeSearchDataset(data[int(len(data) * self.config.train_split):])
        elif stage == "test":
            # self.test_dataset = CodeSearchDataset(self.config.test_dataset)
            raise NotImplementedError("Test dataset not implemented yet.")

    def _load_data_from_jsonl(self, file_paths: list[str]) -> pd.DataFrame:
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
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.batch_size,
                          shuffle=True,
                          collate_fn=self.collator,)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.batch_size,
                          collate_fn=self.collator,)
