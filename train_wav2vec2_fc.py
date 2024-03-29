from transformers import Wav2Vec2CTCTokenizer

from data_utils.dataset import AudioDataset
from training_utils.trainer import Trainer
from models.wav2vec2_fc import Wav2Vec2FC
from utils.logging_utils import setup_logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--num-workers", type=int, required=True)
parser.add_argument("--checkpoint-path", type=str, required=True)

args = parser.parse_args()

logger = setup_logger()

logger.info("Loading tokenizer")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")

logger.info("Defining model")
model = Wav2Vec2FC(
    pretrained_name="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    tokenizer=tokenizer
)

logger.info("Loading dataset")
train_dataset = AudioDataset(
    data_files=["dataset/parquet-train-00000-of-00002.arrow", "dataset/parquet-train-00001-of-00002.arrow"],
    tokenizer=tokenizer
)
test_dataset = AudioDataset(
    data_files=["dataset/parquet-test.arrow"],
    tokenizer=tokenizer
)

logger.info("Defining the trainer")
trainer = Trainer(
    train_dataset=train_dataset,
    dev_dataset=None,
    test_dataset=test_dataset,
    model=model,
    tokenizer=tokenizer,
    batch_size=args.batch_size,
    num_wokers=args.num_workers,
    checkpoint_path=args.checkpoint_path
)

trainer.start()
