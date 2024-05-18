# Project Name

This project fine-tunes transformer-based encoders on a given dataset, evaluates the model's performance, and generates embeddings for the documents in the dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)

## Installation

To use this project, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- Transformers
- Baseline Text Graphs
- finetuning_encoders

### Poetry

This project uses Poetry for dependency management. Follow these steps to set up the environment:

1. **Download from bash:**

    ```bash
    poetry add git+https://github.com/ardaaras99/finetuning-encoders.git@main
    ```

2. **Define as requirement in .toml file:**

    ```toml
    baseline-text-graphs = {git = "https://github.com/ardaaras99/finetuning-encoders.git@main", rev = "main"}
    ```

## Usage

Check the main.py for example usage.

## Parameters

- `TuneableParameters`: Contains parameters for tuning the model.
- `TransformedDataset`: Example dataset to use.
- `RequiredDatasetFormat`: Converts the dataset to the required format.
- `TransformerInput`: Defines the input for the transformer model.
- `EncoderModel`: Defines the transformer-based model.
- `Engine`: Handles the training pipeline.
- `Generator`: Generates embeddings.
- `Evaluator`: Evaluates the model performance.

## What's Next?

1. implementing a structure for downloading and converting the GLUE Benchmark to `RequiredDatasetFormat`.
2. Advanced hyperparameter tuning and visualizations supported with `WandB`.
