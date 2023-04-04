import torch
from torch.utils import data
from transformers import PreTrainedTokenizerBase
import datasets
from tqdm import tqdm
import numpy as np
import os
from typing import List
from utils.instance import Instance

from data_utils.utils import preprocessing_transcript

class AudioDataset(data.Dataset):
    def __init__(self, data_files: List[str], tokenizer: PreTrainedTokenizerBase, cache_folder: str = ".cache") -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.cache_folder = cache_folder
        if not os.path.isdir(self.cache_folder):
            os.mkdir(self.cache_folder)

        data = []
        for data_file in data_files:
            data.append(datasets.Dataset.from_file(data_file))

        self.__data = {}
        self.__ids = []
        self.max_transcript_len = 0
        id = -1
        for datum in data:
            for item in tqdm(datum, desc="Extracting data"):
                audio = item["audio"]
                self.__data[id] = {
                    "path": audio["path"],
                    "transcript": item["transcription"],
                    "sampling_rate": audio["sampling_rate"]
                }
                if self.max_transcript_len < len(item["transcription"]):
                    self.max_transcript_len = len(item["transcription"])
                id += 1
                self.__ids.append(id)
                if not os.path.isfile(os.path.join(self.cache_folder, f"{id}.npy")):
                    feature = audio["array"]
                    np.save(os.path.join(cache_folder, f"{id}.npy"), feature)

    def __len__(self):
        return len(self.__ids)
    
    def __load_features(self, id):
        file_name = os.path.join(self.cache_folder, f"{id}.npy")
        features = np.load(file_name)

        return torch.tensor(features).unsqueeze(0)
    
    def __getitem__(self, index: int):
        id = self.__ids[index]
        features = self.__load_features(id)
        
        transcript = self.__data[id]["transcript"]
        transcript = preprocessing_transcript(transcript)
        tokens = self.tokenizer(
                        text=transcript,
                        max_length=self.max_transcript_len,
                        padding=True,
                        return_tensors="pt")["input_ids"]

        sampling_rate = self.__data[id]["sampling_rate"]
        sampling_rate = torch.tensor([sampling_rate])

        return Instance(
            features=features,
            tokens=tokens
        )
