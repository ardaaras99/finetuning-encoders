import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetuning_encoders.utils import move_device


class Trainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Batch Training", leave=False):
            *inputs, labels = batch  # Assuming the last element in batch is labels
            total_loss += self.run_batch(tuple(inputs), labels)
        return total_loss

    def run_batch(self, inputs: tuple, y: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        inputs = move_device(self.device, *inputs)
        y = y.to(self.device)

        logits = self.model(*inputs)
        loss = F.nll_loss(logits, y.reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()
