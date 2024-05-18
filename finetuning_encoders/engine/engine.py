from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetuning_encoders.engine.evaluator import Evaluator
from finetuning_encoders.engine.feedback import Feedback
from finetuning_encoders.engine.trainer import Trainer


class Engine:
    def __init__(
        self,
        dataset_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loaders: Dict[str, DataLoader],
    ):
        self.trainer = Trainer(model.to(device), optimizer, device)
        self.feedback = Feedback()
        self.evaluator = Evaluator(dataset_name=dataset_name, device=device)
        self.loaders = loaders

        self.device = device

    def pipeline(self, max_epochs: int, patience: int):
        t = tqdm(range(max_epochs), leave=True, desc="Epochs")
        for epoch in t:
            e_loss = self.trainer.train_epoch(self.loaders["train"])
            train_m_d = self.get_metrics("train")
            test_m_d = self.get_metrics("test")

            metric_value = list(test_m_d.values())[0]

            if self.feedback.update(self.trainer.model, metric_value, patience):
                print(f"\nEarly stopping at epoch {epoch}")
                break
            self.feedback.set_descp(t, e_loss, train_m_d, test_m_d, patience)

    def get_metrics(self, loader_key):
        return self.evaluator.eval(
            self.trainer.model, self.device, self.loaders[loader_key]
        )
