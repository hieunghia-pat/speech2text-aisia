import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data
from transformers import PreTrainedTokenizerBase

from tqdm import tqdm
import os

from data_utils.dataset import AudioDataset
from utils.logging_utils import setup_logger
from data_utils.utils import collate_fn
from training_utils.metrics import computes

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = setup_logger()

class Trainer:
    def __init__(self,
                    train_dataset: AudioDataset,
                    dev_dataset: AudioDataset,
                    test_dataset: AudioDataset,
                    model: nn.Module,
                    tokenizer: PreTrainedTokenizerBase,
                    batch_size: int,
                    num_wokers: int,
                    checkpoint_path: str) -> None:
        
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint_path

        if not os.path.isdir(checkpoint_path):
            logger.info("Creating checkpoint path")
            os.mkdir(checkpoint_path)

        logger.info("Creating dataloaders")
        self.train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_wokers
        )
        if self.dev_dataset is not None:
            self.dev_dataloader = data.DataLoader(
                self.dev_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_wokers
            )
        else:
            self.dev_dataloader = None
        self.test_dataloader = data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_wokers
        )

        logger.info("Defining optimization")
        self.optim = Adam(self.model.parameters(), lr=0.01)

        logger.info("Defining loss function")
        self.loss_fn = nn.CTCLoss(
            blank=tokenizer.pad_token_id,
            reduction="mean",
            zero_infinity=True
        )

    def save_checkpoint(self, path: str):
        pass

    def load_checkpoint(self, path: str):
        pass

    def train(self):
        total_loss = 0.
        self.model.train()
        with tqdm(self.train_dataloader, desc=f"Epoch {self.epoch+1} - Training") as pb:
            for ith, item in enumerate(pb):
                item = item.to(device)
                output = self.model(item.features)
                output_len = torch.zeros((output.shape[0], )).fill_(output.shape[1])
                self.optim.zero_grad()
                loss = self.loss_fn(output, item.tokens, item.tokens_len, output_len)
                loss.backward()
                self.optim.step()

                total_loss += loss
                pb.set_postfix({
                    "loss": total_loss / (ith+1)
                })
                pb.update()

    def validate(self):
        self.model.eval()
        with tqdm(self.train_dataloader, desc=f"Epoch {self.epoch+1} - Validating") as pb:
            for ith, item in enumerate(pb):
                item = item.to(device)
                output = self.model(item.features)
                
                pb.set_postfix({
                    "loss": total_loss / (ith+1)
                })
                pb.update()

    def evaluate(self):
        pass

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            state_dict = torch.load(os.path.join(self.checkpoint_path, "last_model.pth"))
            self.model.load_state_dict(state_dict["model"])
            self.optim.load_state_dict(state_dict["optim"])
            self.epoch = state_dict["epoch"]
            self.best_cer = state_dict["best_cer"]
            self.best_wer = state_dict["best_wer"]
        else:
            self.epoch = 0
            self.best_cer = 1.
            self.best_wer = 1.

        while True:
            logger.info(f"Epoch {self.epoch+1}")

            save_best_model = False

            self.train()

            if self.dev_dataloader is not None:
                dev_scores = self.validate()
                logger.info("Validation results: ", dev_scores)
            else:
                dev_scores = None

            test_scores = self.evaluate()
            logger.info("Evaluation results: ", test_scores)
            
            scores = dev_scores if dev_scores is not None else test_scores
            cer = scores["cer"]
            wer = scores["wer"]
            if wer < self.best_wer:
                self.best_cer = cer
                self.best_wer = wer
                save_best_model = True

            if save_best_model:
                self.save_checkpoint("best_model.pth")

            self.save_checkpoint("last_model.path")

            self.epoch += 1