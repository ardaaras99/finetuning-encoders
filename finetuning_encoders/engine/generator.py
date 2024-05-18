from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetuning_encoders.model import EncoderModel
from finetuning_encoders.utils import move_device


class Generator:
    def __init__(
        self,
        model: EncoderModel,
        loaders: Dict[str, DataLoader],
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.loaders = loaders

    def generate_embeddings(self) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            cls_embeddings = []
            for batch in tqdm(self.loaders["full"], desc="Generating Embeddings"):
                *inputs, labels = batch
                inputs = move_device(self.device, *inputs)
                cls_token_embd = self.model.transformer(*inputs)[0][:, 0]
                cls_embeddings.append(cls_token_embd)
            cls_embeddings = torch.cat(cls_embeddings, axis=0)
        return cls_embeddings
