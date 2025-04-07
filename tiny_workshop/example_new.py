import typer
from lib.tiny_voice import data_pipeline, load_model, setup_training_args, train_model

app = typer.Typer()

@app.command()
def run(
    dataset_choice: int = typer.Option(
        None, 
        help="Choose dataset: 1=isixhosa, 2=isizulu, 3=swahili"
    ),
    peft_choice: int = typer.Option(
        None, 
        help="Choose fine-tuning method: 1=partial, 2=lora, 3=ia3"
    ),
):
    # Mapping dictionaries
    dataset_mapping = {
        1: "isixhosa",
        2: "isizulu",
        3: "swahili"
    }
    peft_mapping = {
        1: "partial",
        2: "lora",
        3: "ia3"
    }

    # Prompt for numeric choice if not provided on the command line
    if dataset_choice is None:
        dataset_choice = int(typer.prompt("Enter dataset:\n   1 = isixhosa\n   2 = isizulu\n   3 = swahili\n"))
    
    # Validate dataset selection
    if dataset_choice not in dataset_mapping:
        typer.echo(f"Invalid dataset choice '{dataset_choice}'. Choose 1, 2, or 3.")
        raise typer.Exit(code=1)
    
    # Prompt for numeric choice if not provided on the command line
    if peft_choice is None:
        peft_choice = int(typer.prompt("Enter fine-tuning method:\n   1 = partial,\n   2 = lora,\n   3 = ia3\n"))
    
    # Validate peft selection
    if peft_choice not in peft_mapping:
        typer.echo(f"Invalid PEFT method choice '{peft_choice}'. Choose 1, 2, or 3.")
        raise typer.Exit(code=1)

    # Convert numeric choice to the actual string value
    dataset = dataset_mapping[dataset_choice]
    peft = peft_mapping[peft_choice]

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
