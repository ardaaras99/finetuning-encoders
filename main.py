import torch
from baseline_text_graphs.transformed_dataset import TransformedDataset

from finetuning_encoders.configs import (
    DataLoaderConfig,
    TokenizerConfig,
    TuneableParameters,
)
from finetuning_encoders.dataset import RequiredDatasetFormat, TransformerInput
from finetuning_encoders.engine.engine import Engine
from finetuning_encoders.engine.evaluator import Evaluator
from finetuning_encoders.engine.generator import Generator
from finetuning_encoders.model import EncoderModel
from finetuning_encoders.utils import get_device, modify_tensor

# Parameters for tuning with initial values
params = TuneableParameters()
# The following dataset is for an example, you can replace it with your own dataset and convert it to the required format
d = TransformedDataset(dataset_name=params.dataset_name)

device = get_device()

# Convertion your own dataset to the required format

dataset = RequiredDatasetFormat(
    dataset_name=params.dataset_name,
    documents=d.docs,
    train_mask=modify_tensor(d.train_mask, params.train_percentage),
    test_mask=d.test_mask,
    labels=d.y,
)

# Define the transformer input

inputt = TransformerInput(
    model_checkpoint=params.model_checkpoint,
    dataset=dataset,
    tokenizer_config=TokenizerConfig(max_length=params.max_length),
    data_loader_config=DataLoaderConfig(batch_size=params.batch_size),
)

# Define Model and Optimizer

model = EncoderModel(
    model_checkpoint=params.model_checkpoint,
    dropout=params.dropout,
    n_class=d.n_class,
)

optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

# Define Engine
engine = Engine(
    dataset_name=params.dataset_name,
    model=model,
    optimizer=optimizer,
    device=device,
    loaders=inputt.data_loaders,
)

engine.pipeline(max_epochs=3, patience=1)


# Generate embeddings

generator = Generator(
    model=engine.feedback.best_model,
    loaders=inputt.data_loaders,
    device=device,
)

cls_embds = generator.generate_embeddings()

if cls_embds.shape[0] != len(dataset.documents):
    raise ValueError("Number of documents and embeddings do not match")
else:
    print("We successfully generated embeddings for all documents")

# Test best model final performance for insurance
evaluator = Evaluator(dataset_name=params.dataset_name, device=device)
test_dict = evaluator.eval(
    engine.feedback.best_model, device, inputt.data_loaders["test"]
)
metric_value = list(test_dict.values())[0]

if metric_value == engine.feedback.best_metric:
    print("Result after testing best model:", metric_value)
    print("Last recorded best metric from engine: ", engine.feedback.best_metric)
    print("We successfully tested the best model")
