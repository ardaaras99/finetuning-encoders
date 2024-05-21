# %%
import torch
from baseline_text_graphs.transformed_dataset import TransformedDataset

from finetuning_encoders.configs import (
    DataLoaderConfig,
    TokenizerConfig,
    TuneableParameters,
)
from finetuning_encoders.datamodule import RequiredDatasetFormat, TransformerInput
from finetuning_encoders.engine.engine import Engine
from finetuning_encoders.engine.generator import Generator
from finetuning_encoders.glue.downloader import RawGLUE
from finetuning_encoders.model import EncoderModel
from finetuning_encoders.utils import get_device, modify_tensor, set_seeds

# Parameters for tuning with initial values
dataset_name = "mr"
set_seeds(42)
if dataset_name in ["cola", "sst"]:
    d = RawGLUE(dataset_name=dataset_name)
    documents = d.documents

elif dataset_name in ["mr", "R8", "R52", "ohsumed", "20ng"]:
    d = TransformedDataset(dataset_name=dataset_name)
    documents = d.docs

params = TuneableParameters(
    dataset_name=dataset_name,
    train_percentage=10,
    max_length=64,
    dropout=0.3,
    lr=1e-5,
    batch_size=32,
    model_checkpoint="roberta-base",
)
train_mask = modify_tensor(d.train_mask, params.train_percentage)
dataset = RequiredDatasetFormat(
    dataset_name=params.dataset_name,
    documents=documents,
    train_mask=train_mask,
    test_mask=d.test_mask,
    labels=d.y,
)

# Define the transformer input
# %%
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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params.lr,
    weight_decay=0.001,
)

# Define Engine
engine = Engine(
    dataset_name=params.dataset_name,
    model=model,
    optimizer=optimizer,
    device=get_device(),
    loaders=inputt.data_loaders,
)

engine.pipeline(max_epochs=10, patience=2)


# Generate embeddings

generator = Generator(
    model=engine.best_model,
    loader=inputt.data_loaders,
    device=get_device(),
)

cls_embds = generator.generate_embeddings()

if cls_embds.shape[0] != len(dataset.documents):
    raise ValueError("Number of documents and embeddings do not match")
else:
    print("We successfully generated embeddings for all documents")


# Test best model final performance for insurance
# %%
engine.model = engine.best_model
test_dict = engine.eval("test")
metric_value = list(test_dict.values())[0]
if metric_value == engine.best_metric_val:
    print("Result after testing best model:", metric_value)
    print("Last recorded best metric from engine: ", engine.best_metric_val)
    print("We successfully tested the best model")

# %%
# At the end we need to save the embeddings, best_model and the train mask
