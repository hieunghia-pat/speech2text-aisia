import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from typing import List
from tqdm import tqdm

from data_utils.dataset import AudioDataset
from utils.logging_utils import setup_logger

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = setup_logger()

class Trainer:
    def __init__(self,
                    train_dataset: AudioDataset,
                    dev_dataset: AudioDataset,
                    test_dataset: AudioDataset,
                    model: nn.Module,
                    tokenizer: PreTrainedTokenizerBase) -> None:
        
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.model = model.to(device)
        self.tokenizer = tokenizer

    def train(self):
        pass

    def evaluate(self):
        pass
