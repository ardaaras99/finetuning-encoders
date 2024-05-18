import os
from pathlib import Path

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

# Set TOKENIZERS_PARALLELISM to false to suppress the warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# An example of a dictionary that maps dataset names to metric names
DATASET_TO_METRIC_NAME = {
    "mr": "acc",
    "sst2": "acc",
    "cola": "mcc",
    "R8": "acc",
    "R52": "acc",
    "ohsumed": "acc",
}


PROJECT_PATH = Path.cwd()
