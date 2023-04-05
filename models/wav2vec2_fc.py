import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, PreTrainedTokenizerBase

class Wav2Vec2FC(nn.Module):
    def __init__(self, pretrained_name: str, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()

        self.wav2vec = AutoModel.from_pretrained(pretrained_name)
        # freeze the base model
        for param in self.wav2vec.parameters():
            param.requires_grad_ = False
        
        # layers for fine-tuning
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, tokenizer.vocab_size)

    def forward(self, features: torch.Tensor):
        features = self.wav2vec(features).last_hidden_state
        features = self.fc(self.dropout(features))

        return F.log_softmax(features, dim=-1)
