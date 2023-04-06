from jiwer import cer, wer
from typing import List
import numpy as np

def computes(hypos: List[str], refs: List[str]):
    cers = []
    wers = []
    for hypo, ref in zip(hypos, refs):
        cers.append(cer(ref, hypo))
        wers.append(wer(ref, hypo))

    return {
        "cer": np.array(cers).mean(),
        "wer": np.array(wers).mean()
    }