import pickle

import datasets
import torch

from finetuning_encoders import PROJECT_PATH

GLUE_DATASET_PATH = PROJECT_PATH.joinpath("data/glue-datasets")


class LabelEncoder:
    label_to_int = {}

    def __init__(self, raw_labels):
        self.raw_labels = raw_labels
        self.y = self._create_label_mapping()

    def _create_label_mapping(self):
        sorted_labels = sorted(set(self.raw_labels))
        LabelEncoder.label_to_int = {label: i for i, label in enumerate(sorted_labels)}
        return [LabelEncoder.label_to_int[label] for label in self.raw_labels]


class RawGLUE:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        GLUE_DATASET_PATH.mkdir(parents=True, exist_ok=True)
        self.dataset_path = GLUE_DATASET_PATH.joinpath(f"{dataset_name}_raw_data.pkl")

        if self.dataset_path.is_file():
            print(f"{dataset_name} dataset already exists.")
        else:
            self.downnload()
        self.convert_to_required_format()

    def downnload(self) -> None:
        dataset = datasets.load_dataset("glue", self.dataset_name)
        pickle.dump(dataset, open(self.dataset_path, "wb"))

    def convert_to_required_format(self):
        raw_dataset = pickle.load(open(self.dataset_path, "rb"))

        train = raw_dataset["train"].to_pandas()
        test = raw_dataset["test"].to_pandas()

        self.documents = train["sentence"].tolist() + test["sentence"].tolist()
        train_labels = train["label"].tolist()
        test_labels = test["label"].tolist()
        train_size = len(train_labels)
        test_size = len(test_labels)

        self.raw_labels = train_labels + test_labels
        self.label_encoder = LabelEncoder(self.raw_labels)
        self.y = torch.tensor(self.label_encoder.y).reshape(-1, 1)

        self.train_mask = torch.tensor([1] * train_size + [0] * test_size).reshape(-1)
        self.test_mask = torch.tensor([0] * train_size + [1] * test_size).reshape(-1)
        self.int_to_label = {v: k for k, v in self.label_encoder.label_to_int.items()}
        self.n_class = len(list(set(self.raw_labels)))
