import copy
from typing import Dict

from tqdm.auto import tqdm


class Feedback:
    def __init__(self):
        self.best_model = None
        self.best_metric = 0.0
        self.counter = 0

    def update(self, model, metric_value: float, patience: int) -> bool:
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_model = copy.deepcopy(model)
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= patience

    def set_descp(
        self,
        t: tqdm,
        e_loss: float,
        train_m_d: Dict[str, float],
        test_m_d: Dict[str, float],
        patience: int,
    ):
        metric_name = list(train_m_d.keys())[0]
        t.set_description(
            f"Loss: {e_loss:.4f}"
            + f" Best Test {metric_name}: {self.best_metric:.3f},"
            + f" Train {metric_name}: {train_m_d[metric_name]:.3f},"
            + f" Test {metric_name}: {test_m_d[metric_name]:.3f}"
            + f" Patience Counter: {self.counter}/{patience}"
        )
