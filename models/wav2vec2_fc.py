import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCTC, PreTrainedTokenizerBase

class Wav2Vec2FC(nn.Module):
    def __init__(self, pretrained_name: str, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()

        pretrained_model = AutoModelForCTC.from_pretrained(
                                            pretrained_name,
                                            ctc_loss_reduction="mean",
                                            pad_token_id=tokenizer.pad_token_id)
        self.wav2vec = pretrained_model.wav2vec2
        
        # layers for fine-tuning
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, tokenizer.vocab_size)

    def forward(self, features: torch.Tensor):
        features = self.wav2vec(features, output_hidden_states=True).hidden_states[-1]
        features = self.fc(self.dropout(features))

        return F.log_softmax(features, dim=-1)
