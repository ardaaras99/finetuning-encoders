import copy
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetuning_encoders import DATASET_TO_METRIC_NAME


class Engine:
    def __init__(
        self,
        dataset_name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loaders: Dict[str, DataLoader],
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loaders = loaders
        self.dataset_name = dataset_name

        self.device = device
        self.best_model = None
        self.best_metric_val = None
        self.counter = 0

    def pipeline(self, max_epochs: int, patience: int):
        t = tqdm(range(max_epochs), leave=True, desc="Epochs")
        for _ in t:
            e_loss = self.fit()
            train_m_d = self.eval("train")
            test_m_d = self.eval("test")
            metric_value = list(test_m_d.values())[0]

            if self.best_metric_val is None or metric_value > self.best_metric_val:
                self.best_metric_val = metric_value
                self.best_model = copy.deepcopy(self.model)
                self.counter = 0
            else:
                self.counter += 1

            metric_name = DATASET_TO_METRIC_NAME[self.dataset_name]
            t.set_description(
                f"Loss: {e_loss:.4f}"
                + f" Best Test {metric_name}: {self.best_metric_val:.3f},"
                + f" Train {metric_name}: {train_m_d[metric_name]:.3f},"
                + f" Test {metric_name}: {test_m_d[metric_name]:.3f}"
                + f" Patience Counter: {self.counter}/{patience}"
            )

            if self.counter >= patience:
                break

    def eval(self, split: str):
        self.model.eval()
        y_pred, y = [], []
        with torch.no_grad():
            t = tqdm(self.loaders[split], desc="Evaluation", leave=False)
            for i_ids, a_mask, labels in t:
                i_ids, a_mask, labels = (
                    i_ids.to(self.device),
                    a_mask.to(self.device),
                    labels.to(self.device),
                )

                y_pred.append(self.model(i_ids, a_mask).to("cpu"))
                y.append(labels)

        y = torch.cat(y, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        return self.compute_metrics(y_pred, y)

    def compute_metrics(
        self, y_pred: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, float]:
        y_pred = y_pred.argmax(dim=1).cpu().numpy()
        y = y.cpu().numpy()

        metric_name = DATASET_TO_METRIC_NAME[self.dataset_name]

        if metric_name == "acc":
            return {"acc": 100 * accuracy_score(y, y_pred)}
        elif metric_name == "mcc":
            return {"mcc": 100 * matthews_corrcoef(y, y_pred)}
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")

    def fit(self):
        self.model.train()
        epoch_loss = 0.0
        t = tqdm(self.loaders["train"], desc="Batch Training", leave=False)
        for i_ids, a_mask, labels in t:
            loss = self.run_batch(i_ids, a_mask, labels)
            epoch_loss += loss

        return epoch_loss

    def run_batch(
        self, i_ids: torch.Tensor, a_mask: torch.Tensor, labels: torch.Tensor
    ):
        self.optimizer.zero_grad()

        i_ids, a_mask, labels = (
            i_ids.to(self.device),
            a_mask.to(self.device),
            labels.to(self.device),
        )

        logits = self.model(i_ids, a_mask)

        loss = F.nll_loss(logits, labels.reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()
