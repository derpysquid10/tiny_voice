import typer
from lib.tiny_voice import data_pipeline, load_model, setup_training_args, train_model
from datetime import datetime

app = typer.Typer()

@app.command()
def run(
    dataset: str = typer.Option(None, help="Dataset to use: isixhosa, isizulu, or swahili"),
    peft: str = typer.Option(None, help="Fine-tuning type: partial, lora, or ia3"),
):
    # -----------------------------------------------------
    # Get user input for dataset and PEFT method
    # -----------------------------------------------------
    if dataset is None:
        dataset = typer.prompt("Enter dataset (isixhosa, isizulu, swahili)").strip().lower()
    if peft is None:
        peft = typer.prompt("Enter fine-tuning method (partial, lora, ia3)").strip().lower()

    # Validate
    valid_datasets = {"isixhosa", "isizulu", "swahili"}
    valid_peft = {"partial", "lora", "ia3"}

    if dataset not in valid_datasets:
        typer.echo(f"Invalid dataset '{dataset}'. Choose from: {', '.join(valid_datasets)}")
        raise typer.Exit(code=1)

    if peft not in valid_peft:
        typer.echo(f"Invalid PEFT method '{peft}'. Choose from: {', '.join(valid_peft)}")
        raise typer.Exit(code=1)

    # ----------------------------------------------------
    # Below is an example of using tiny voice to train a model
    # ----------------------------------------------------

    # Load dataset and processor
    data, processor = data_pipeline(dataset)

    # Load and configure model
    model = load_model(peft)

    # Training arguments
    training_args = setup_training_args(peft)

    # Train
    train_model(model, data, processor, peft, training_args)

if __name__ == "__main__":
    app()
