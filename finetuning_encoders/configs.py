from dataclasses import dataclass


@dataclass
class TuneableParameters:
    dataset_name: str = "mr"
    model_checkpoint: str = "roberta-base"
    max_length: int = 8
    batch_size: int = 32
    dropout: float = 0.1
    lr: float = 5e-5
    train_percentage: int = 10


@dataclass
class TokenizerConfig:
    max_length: int
    truncation: bool = True
    padding: bool = True
    return_tensors: str = "pt"


@dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
