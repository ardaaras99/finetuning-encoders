from finetuning_encoders import PROJECT_PATH
from finetuning_encoders.configs import DataLoaderConfig, TokenizerConfig
from finetuning_encoders.datamodule import RequiredDatasetFormat, TransformerInput
from finetuning_encoders.engine.engine import Engine
from finetuning_encoders.engine.generator import Generator
from finetuning_encoders.utils import get_device, get_raw_data, read_best_model


def main(model_checkpoint, dataset_name, train_percentage):

    best_model, train_mask, saved_test_acc, max_length = read_best_model(
        model_checkpoint=model_checkpoint,
        dataset_name=dataset_name,
        train_percentage=train_percentage,
    )

    d, documents = get_raw_data(dataset_name)
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
        tokenizer_config=TokenizerConfig(max_length=max_length),
        data_loader_config=DataLoaderConfig(batch_size=32),
    )

    generator = Generator(
        model=best_model,
        loader=inputt.data_loaders["full"],
        device=get_device(),
    )

    engine = Engine(
        dataset_name=dataset_name,
        model=best_model,
        optimizer=None,
        device=get_device(),
        loaders=inputt.data_loaders,
    )

    test_dict = engine.eval("test")

    metric_value = list(test_dict.values())[0]
    Generator.test_accs_match(metric_value, saved_test_acc)

    cls_embds = generator.generate_embeddings()
    Generator.test_embeddings(cls_embds, len(dataset.documents))
    Generator.save_embeddings(
        cls_embds, dataset_name, train_percentage, model_checkpoint
    )


if __name__ == "__main__":
    for dataset_name in ["mr", "R8", "R52", "ohsumed", "sst2", "cola"]:
        for train_percentage in [1, 5, 10, 20]:
            saving_convention = (
                f"best_models/roberta-base-{dataset_name}-{train_percentage}"
            )
            path = PROJECT_PATH.joinpath(saving_convention)
            if path.exists():
                print("Running generation for:", dataset_name, train_percentage)
                main("roberta_base", dataset_name, train_percentage)
