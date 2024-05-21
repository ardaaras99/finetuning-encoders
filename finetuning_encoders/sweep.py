from os import getenv
from pathlib import Path

import torch

import wandb
from finetuning_encoders import DATASET_TO_METRIC_NAME
from finetuning_encoders.configs import DataLoaderConfig, TokenizerConfig
from finetuning_encoders.datamodule import RequiredDatasetFormat, TransformerInput
from finetuning_encoders.engine.engine import Engine
from finetuning_encoders.model import EncoderModel
from finetuning_encoders.utils import get_device, get_raw_data, modify_tensor, set_seeds

MODEL_CHECKPOINT = None
DATASET_NAME = None
TRAIN_PERCENTAGE = None
MONITOR_METRIC = None
GLOBAL_BEST_METRIC = 0.0
CKPT_FOLDER_FOR_SWEEP = None


def generate_sweep(
    model_checkpoint: str,
    dataset_name: str,
    train_percentage: float,
    project: str = None,
) -> str:
    global MODEL_CHECKPOINT, DATASET_NAME, TRAIN_PERCENTAGE

    MODEL_CHECKPOINT = model_checkpoint
    DATASET_NAME = dataset_name
    TRAIN_PERCENTAGE = train_percentage
    MONITOR_METRIC = f"test/{DATASET_TO_METRIC_NAME[DATASET_NAME]}"

    if project is None:
        project = getenv("WANDB_PROJECT")

    sweep_configuration = {
        "name": f"{MODEL_CHECKPOINT}-{DATASET_NAME}-{TRAIN_PERCENTAGE}",
        "method": "random",
        "metric": {
            "goal": "maximize",
            "name": MONITOR_METRIC,
        },
        "parameters": {
            "model_checkpoint": {"value": MODEL_CHECKPOINT},
            "dataset_name": {"value": DATASET_NAME},
            "monitor_metric": {"value": MONITOR_METRIC},
            "train_percentage": {"value": TRAIN_PERCENTAGE},
            "max_epochs": {"value": 10},
            "patience": {"value": 3},
            "batch_size": {"values": [16, 32, 64]},
            "max_seq_length": {"values": [32, 64, 128]},
            "learning_rate": {"min": 5e-6, "max": 5e-4},
            "dropout": {"min": 0.1, "max": 0.5},
        },
    }
    sweep_id = wandb.sweep(sweep_configuration, project=project)
    return sweep_id


def add_agent(sweep_id: str, entity: str = None, project: str = None) -> None:
    global MODEL_CHECKPOINT, DATASET_NAME, TRAIN_PERCENTAGE, MONITOR_METRIC, CKPT_FOLDER_FOR_SWEEP

    if entity is None:
        entity = getenv("WANDB_ENTITY")
    if project is None:
        project = getenv("WANDB_PROJECT")
    if entity is None or project is None:
        raise ValueError("Must specify entity and project.")

    tuner = wandb.controller(sweep_id, entity=entity, project=project)
    parameters = tuner.sweep_config.get("parameters")
    if parameters is not None:
        MODEL_CHECKPOINT = parameters.get("model_checkpoint")["value"]
        DATASET_NAME = parameters.get("dataset_name")["value"]
        TRAIN_PERCENTAGE = parameters.get("train_percentage")["value"]
        MONITOR_METRIC = parameters.get("monitor_metric")["value"]

    CKPT_FOLDER_FOR_SWEEP = Path(
        f"best_models/{MODEL_CHECKPOINT}-{DATASET_NAME}-{TRAIN_PERCENTAGE}"
    )
    CKPT_FOLDER_FOR_SWEEP.mkdir(parents=True, exist_ok=True)

    wandb.agent(sweep_id, function=train, count=40)


def train() -> None:
    global GLOBAL_BEST_METRIC, CKPT_FOLDER_FOR_SWEEP

    wandb.init()
    config = wandb.config
    model_checkpoint = config.model_checkpoint
    dataset_name = config.dataset_name
    train_percentage = config.train_percentage
    monitor_metric = config.monitor_metric
    max_epochs = config.max_epochs
    patience = config.patience
    batch_size = config.batch_size
    max_seq_length = config.max_seq_length
    learning_rate = config.learning_rate
    dropout = config.dropout

    set_seeds(seed_no=42)
    d, documents = get_raw_data(dataset_name)
    train_mask = modify_tensor(d.train_mask, train_percentage)

    dataset = RequiredDatasetFormat(
        dataset_name=dataset_name,
        documents=documents,
        train_mask=train_mask,
        test_mask=d.test_mask,
        labels=d.y,
    )

    inputt = TransformerInput(
        model_checkpoint=model_checkpoint,
        dataset=dataset,
        tokenizer_config=TokenizerConfig(max_length=max_seq_length),
        data_loader_config=DataLoaderConfig(batch_size=batch_size),
    )

    model = EncoderModel(
        model_checkpoint=model_checkpoint, n_class=d.n_class, dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loaders = inputt.data_loaders

    engine = Engine(
        dataset_name=dataset_name,
        model=model,
        optimizer=optimizer,
        device=get_device(),
        loaders=loaders,
    )
    engine.pipeline(max_epochs=max_epochs, patience=patience)
    best_test_metric = engine.best_metric_val

    if best_test_metric > GLOBAL_BEST_METRIC:
        # delete previous best model
        print("Deleting previous best global model for sweep")
        for file in CKPT_FOLDER_FOR_SWEEP.glob("best_model_*.pth"):
            file.unlink()

        GLOBAL_BEST_METRIC = best_test_metric
        saving_convention = (
            f"best_model_{round(best_test_metric,4)}_{max_seq_length}.pth"
        )
        file_path = CKPT_FOLDER_FOR_SWEEP.joinpath(saving_convention)
        torch.save(engine.best_model, file_path)
        torch.save(train_mask, CKPT_FOLDER_FOR_SWEEP.joinpath("train_mask.pth"))

    wandb.log({monitor_metric: best_test_metric})
