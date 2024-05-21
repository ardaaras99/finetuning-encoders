import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetuning_encoders import PROJECT_PATH
from finetuning_encoders.model import EncoderModel
from finetuning_encoders.utils import move_device


class Generator:
    def __init__(self, model: EncoderModel, loader: DataLoader, device: torch.device):
        self.model = model
        self.device = device
        self.loader = loader

    def generate_embeddings(self) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            cls_embeddings = []
            t = tqdm(self.loader, desc="Generating Embeddings")
            for batch in t:
                *inputs, labels = batch
                inputs = move_device(self.device, *inputs)
                cls_token_embd = self.model.transformer(*inputs)[0][:, 0]
                cls_embeddings.append(cls_token_embd)
            cls_embeddings = torch.cat(cls_embeddings, axis=0)
        return cls_embeddings

    @staticmethod
    def test_embeddings(cls_embds: torch.Tensor, n_docs: int) -> None:
        if cls_embds.shape[0] != n_docs:
            raise ValueError("Number of documents and embeddings do not match")
        else:
            print("We successfully generated embeddings for all documents")

    @staticmethod
    def test_accs_match(metric_value: float, saved_test_acc: float) -> None:
        print("Result after testing best model:", round(metric_value, 4))
        print("Saved test accuracy:", round(saved_test_acc, 4))
        if round(metric_value, 4) != round(saved_test_acc, 4):
            raise ValueError("Something went wrong while testing the best model")
        else:
            print("Accuracy comparasion Test passed")

    @staticmethod
    def save_embeddings(cls_embds, dataset_name, train_percentage, model_checkpoint):
        saving_convention = f"{model_checkpoint}-{dataset_name}-{train_percentage}"
        folder_path = PROJECT_PATH.joinpath(f"best_models/{saving_convention}")
        torch.save(cls_embds, folder_path.joinpath("embeddings.pth"))
