import re
from typing import List
from utils.instance import InstanceList, Instance

def preprocessing_transcript(transcript: str) -> str:
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    transcript = re.sub(chars_to_ignore_regex, "", transcript)
    transcript = transcript.lower()

    return transcript

def pad_tokens(tokens: List[str], max_len: int, pad_value: str):
    if len(tokens) >= max_len:
        return tokens
    
    delta_len = max_len - len(tokens)
    tokens.extend([pad_value]*delta_len)

    return tokens

def collate_fn(instances: List[Instance]) -> InstanceList:
    return InstanceList(instances)