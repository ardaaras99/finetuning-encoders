import click
from dotenv import load_dotenv

from finetuning_encoders.sweep import add_agent, generate_sweep

load_dotenv()


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("model_checkpoint", type=click.STRING)
@click.argument("dataset_name", type=click.STRING)
@click.argument("train_percentage", type=click.FLOAT)
def sweep(model_checkpoint: str, dataset_name: str, train_percentage: float) -> None:
    "Initialize a WandB sweep for fine-tuning encoder model, generated from MODEL_CHECKPOINT on DATASET_NAME"
    sweep_id = generate_sweep(model_checkpoint, dataset_name, train_percentage)
    print(f"Created sweep with id: {sweep_id}")


@cli.command()
@click.argument("sweep_id", type=click.STRING)
def agent(sweep_id: str) -> None:
    "Attach an agent to the sweep with SWEEP_ID"
    add_agent(sweep_id)


cli()
