import torch
import torchaudio
from torch.utils import data
from transformers import PreTrainedTokenizerBase
import datasets
from tqdm import tqdm
import numpy as np
import os
import json
from typing import List
from utils.instance import Instance

from data_utils.utils import preprocessing_transcript

class AudioDataset(data.Dataset):
    def __init__(self, 
                    data_files: List[str], 
                    tokenizer: PreTrainedTokenizerBase, 
                    cache_folder: str = ".cache", 
                    split: str = "train") -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.cache_folder = cache_folder

        if os.path.isfile(os.path.join(self.cache_folder, f"{split}.json")):
            json_data = json.load(open(os.path.join(self.cache_folder, f"{split}.json"), "r"))
            self.__data = {int(key): value for key, value in json_data["data"].items()}
            self.__ids = json_data["ids"]
            self.max_transcript_len = json_data["max_transcript_len"]
            return

        if not os.path.isdir(os.path.join(self.cache_folder, split)):
            os.makedirs(os.path.join(self.cache_folder, split))

        data = []
        for data_file in data_files:
            data.append(datasets.Dataset.from_file(data_file))

        self.__data = {}
        self.__ids = []
        self.max_transcript_len = 0
        id = -1
        for datum in data:
            for item in tqdm(datum, desc="Getting data"):
                audio = item["audio"]
                id += 1
                self.__data[id] = {
                    "path": audio["path"],
                    "features_path": os.path.join(cache_folder, split, f"{id}.npy"),
                    "transcript": item["transcription"],
                    "sampling_rate": audio["sampling_rate"]
                }
                if self.max_transcript_len < len(item["transcription"]):
                    self.max_transcript_len = len(item["transcription"])
                self.__ids.append(id)
                feature = audio["array"]
                np.save(os.path.join(cache_folder, split, f"{id}.npy"), feature)

        json.dump({
            "data": self.__data,
            "ids": self.__ids,
            "max_transcript_len": self.max_transcript_len
        }, open(os.path.join(self.cache_folder, f"{split}.json"), "w+"), ensure_ascii=False)

    def __len__(self):
        return len(self.__ids)
    
    def __load_features(self, file_name):
        features = np.load(file_name)

        return torch.tensor(features)[:100].unsqueeze(0)
    
    def __getitem__(self, index: int):
        id = self.__ids[index]
        features_path = self.__data[id]["features_path"]
        features = self.__load_features(features_path)
        
        transcript = self.__data[id]["transcript"]
        transcript = preprocessing_transcript(transcript)
        tokens = self.tokenizer(
                        text=transcript,
                        add_special_tokens=False,
                        max_length=self.max_transcript_len,
                        padding="max_length",
                        return_tensors="pt")["input_ids"]
        tokens_len = torch.tensor([len(self.tokenizer.tokenize(transcript))]).unsqueeze(0)

        return Instance(
            features=features,
            tokens=tokens,
            tokens_len=tokens_len
        )
