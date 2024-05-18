from typing import Dict

import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetuning_encoders import DATASET_TO_METRIC_NAME
from finetuning_encoders.model import EncoderModel
from finetuning_encoders.utils import move_device


class Evaluator:
    def __init__(self, dataset_name: str, device: torch.device):
        self.dataset_name = dataset_name
        self.device = device

    def eval(self, model: EncoderModel, device, loader: DataLoader):
        model.eval()
        y_pred, y = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluation", leave=False):
                *inputs, labels = batch
                inputs = move_device(device, *inputs)
                y_pred.append(model(*inputs).to("cpu"))
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
