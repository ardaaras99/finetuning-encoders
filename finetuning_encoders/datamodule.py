from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from finetuning_encoders.configs import DataLoaderConfig, TokenizerConfig


class RequiredDatasetFormat:
    def __init__(
        self,
        dataset_name: str,
        documents: List[str],
        train_mask: torch.Tensor,
        test_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.dataset_name = dataset_name
        self.documents = documents
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.labels = labels
        self.full_mask = torch.tensor([1] * len(documents))


class TransformerInput:
    def __init__(
        self,
        dataset: RequiredDatasetFormat,
        model_checkpoint: str,
        tokenizer_config: TokenizerConfig,
        data_loader_config: DataLoaderConfig,
    ):
        self.dataset = dataset
        self.model_checkpoint = model_checkpoint
        self.tokenizer_config = tokenizer_config
        self.data_loader_config = data_loader_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.tensor_datasets: Dict[str, TensorDataset] = {}
        self.data_loaders: Dict[str, DataLoader] = {}

        self._set_loaders()

    def _set_loaders(self):
        encoded_inputs = self.tokenizer(
            self.dataset.documents,
            truncation=self.tokenizer_config.truncation,
            padding=self.tokenizer_config.padding,
            max_length=self.tokenizer_config.max_length,
            return_tensors=self.tokenizer_config.return_tensors,
        )

        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        z = zip(
            [self.dataset.train_mask, self.dataset.test_mask, self.dataset.full_mask],
            ["train", "test", "full"],
        )

        for mask, split in z:
            self.tensor_datasets[split] = TensorDataset(
                input_ids[mask == 1],
                attention_mask[mask == 1],
                self.dataset.labels[mask == 1],
            )

            self.data_loaders[split] = DataLoader(
                self.tensor_datasets[split],
                batch_size=self.data_loader_config.batch_size,
                shuffle=self.data_loader_config.shuffle,
                num_workers=self.data_loader_config.num_workers,
                pin_memory=self.data_loader_config.pin_memory,
            )

    def get_data_loader(self, split: str) -> DataLoader:
        if split not in self.data_loaders:
            raise ValueError(f"DataLoader for split '{split}' has not been created.")
        return self.data_loaders[split]
