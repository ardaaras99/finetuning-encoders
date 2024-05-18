import os
import pickle
from pathlib import Path

import datasets
from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

# Set TOKENIZERS_PARALLELISM to false to suppress the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_TO_METRIC_NAME = {
    "mr": "acc",
    "sst2": "acc",
    "cola": "mcc",
    "R8": "acc",
    "R52": "acc",
    "ohsumed": "acc",
}


PROJECT_PATH = Path.cwd()
GLUE_DATASET_PATH = PROJECT_PATH.joinpath("data/glue-datasets")
GLUE_DATASET_PATH.mkdir(parents=True, exist_ok=True)

for dataset_name in ["cola", "sst2"]:
    dataset_path = GLUE_DATASET_PATH.joinpath(f"{dataset_name}_raw_data.pkl")
    if dataset_path.is_file():
        print(f"{dataset_name} dataset already exists.")
        continue

    else:
        dataset = datasets.load_dataset("glue", dataset_name)
        pickle.dump(
            dataset,
            open(GLUE_DATASET_PATH.joinpath(f"{dataset_name}_raw_data.pkl"), "wb"),
        )
