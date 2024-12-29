from dataclasses import dataclass
from pyparsing import PrecededBy
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
                 input_dict: dict[str, list]):
        return self.collate(input_dict=input_dict)

    def collate(self,
                input_dict: dict[str,list]):
        collated = {}
        for k,v in input_dict.items():
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
        if stage == "fit":
            self.train_dataset = CodeSearchDataset(self.config.train_dataset)
            self.val_dataset = CodeSearchDataset(self.config.validation_dataset)
        elif stage == "test":
            self.test_dataset = CodeSearchDataset(self.config.test_dataset)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.train_batch_size,
                          shuffle=True,
                          collate_fn=self.collator,
                          num_workers=self.config.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.val_batch_size,
                          collate_fn=self.collator,
                          num_workers=self.config.num_workers)
