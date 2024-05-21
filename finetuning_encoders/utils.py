import glob
import random
import re

import numpy as np
import torch
from baseline_text_graphs.transformed_dataset import TransformedDataset

from finetuning_encoders import PROJECT_PATH
from finetuning_encoders.glue.downloader import RawGLUE


def read_best_model(model_checkpoint: str, dataset_name: str, train_percentage: float):
    saving_convention = f"{model_checkpoint}-{dataset_name}-{train_percentage}"
    file_path = PROJECT_PATH.joinpath(f"best_models/{saving_convention}")
    best_model_files = glob.glob(str(file_path.joinpath("best_model_*.pth")))

    # Check if any files match the pattern
    if not best_model_files:
        raise FileNotFoundError("No model files found matching the pattern")
    else:
        # Load the best model
        best_model_file = best_model_files[0]
        match = re.search(r"best_model_(\d+\.\d+)_(\d+)\.pth", best_model_file)

        if match:
            saved_test_acc = float(match.group(1))
            max_length = int(match.group(2))
            print(f"Extracted test_acc: {saved_test_acc}")
            print(f"Extracted max_length: {max_length}")
        else:
            raise ValueError("Filename does not match the expected pattern")

        best_model = torch.load(best_model_file)
        print(f"Loaded model from {best_model_file}")

    train_mask = torch.load(file_path.joinpath("train_mask.pth"))
    return best_model, train_mask, saved_test_acc, max_length


def get_raw_data(dataset_name: str):

    if dataset_name in ["cola", "sst"]:
        d = RawGLUE(dataset_name=dataset_name)
        documents = d.documents

    elif dataset_name in ["mr", "R8", "R52", "ohsumed", "20ng"]:
        d = TransformedDataset(dataset_name=dataset_name)
        documents = d.docs

    else:
        raise ValueError("Invalid dataset name")

    return d, documents


def get_device():
    if torch.torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    return device


def set_seeds(seed_no: int = 42):
    random.seed(seed_no)
    np.random.seed(seed_no)
    torch.manual_seed(seed_no)
    torch.cuda.manual_seed_all(seed_no)


def move_device(device: torch.device, *args):
    return [x.to(device) for x in args]


def modify_tensor(tensor, percentage):
    set_seeds(42)
    # Ensure the percentage is between 0 and 100
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    # Find the initial number of ones (m)
    one_indices = torch.nonzero(tensor, as_tuple=False).flatten()
    m = one_indices.size(0)

    # Calculate the number of ones to keep
    keep_count = int((percentage / 100) * m)

    # Randomly select keep_count indices
    random_indices = torch.randperm(m)[:keep_count]

    # Create a new tensor to modify
    new_tensor = torch.zeros_like(tensor)

    # Set the randomly selected elements to 1
    new_tensor[one_indices[random_indices]] = 1

    return new_tensor
