import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import AutoModel


class EncoderModel(nn.Module):
    def __init__(self, model_checkpoint, n_class, dropout: float = 0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_checkpoint)
        fan_out = list(self.transformer.modules())[-2].out_features

        self.classifier = nn.Linear(fan_out, n_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.transformer(input_ids, attention_mask)[0][:, 0]
        out = self.dropout(cls_feats)
        logits = self.classifier(out)
        out = F.log_softmax(logits, dim=1)
        return out
