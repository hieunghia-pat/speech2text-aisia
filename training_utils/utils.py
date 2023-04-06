import torch
from transformers import PreTrainedTokenizerBase
import itertools

from typing import List

def decode(x: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> str:
    return tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def batch_decode(batched_x: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> List[str]:
    batched_x = batched_x.tolist()
    decoded_texts = []
    for x in batched_x:
        decoded_text = decode(x, tokenizer)
        decoded_text = "".join([k for k, _ in itertools.groupby(decoded_text)])
        decoded_texts.append(decoded_text)

    return decoded_texts