"""
Download Afrispeech-200 data from huggingface datasets
"""

from datasets import load_dataset, load_from_disk, DatasetDict, Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # <--- Add this line
import matplotlib.pyplot as plt
from tabulate import tabulate
from config import HF_CACHE_DIR, EDA_DIR, PROCESSED_DATA_DIR, MODEL_NAME
import numpy as np
import typer


app = typer.Typer()

def load_data(dataset: str) -> DatasetDict:
    """
    Load Afrispeech-200 data from huggingface datasets, with the isizulu accent from South Africa
    Args:
        dataset (str): The name of the dataset to load. Must be one of "isizulu", “swahili”, “isixhosa”
            "isizulu" for isiZulu accent from South Africa, a medium sized dataset with 3 hours of data
            "swahili" for Swahili accent from Kenya, a larger dataset with 15 hours of data
            "isixhosa" for isiXhosa accent from South Africa, a smaller dataset with only 35 minutes of data
    """
    if dataset not in ["isizulu", "swahili", "isixhosa"]:
        raise ValueError("Dataset must be one of 'isizulu', 'swahili', 'isixhosa'")
    
    afrispeech = DatasetDict()
    if dataset == "isizulu":
        afrispeech["train"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="train", cache_dir=HF_CACHE_DIR)
        afrispeech["val"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="validation", cache_dir=HF_CACHE_DIR)
        afrispeech["test"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="test", cache_dir=HF_CACHE_DIR)
    elif dataset == "swahili":
        afrispeech["train"] = load_dataset("tobiolatunji/afrispeech-200", "swahili", split="train", cache_dir=HF_CACHE_DIR)
        afrispeech["val"] = load_dataset("tobiolatunji/afrispeech-200", "swahili", split="validation", cache_dir=HF_CACHE_DIR)
        afrispeech["test"] = load_dataset("tobiolatunji/afrispeech-200", "swahili", split="test", cache_dir=HF_CACHE_DIR)
    elif dataset == "isixhosa":
        afrispeech["train"] = load_dataset("tobiolatunji/afrispeech-200", "isixhosa", split="train", cache_dir=HF_CACHE_DIR)
        afrispeech["val"] = load_dataset("tobiolatunji/afrispeech-200", "isixhosa", split="validation", cache_dir=HF_CACHE_DIR)
        afrispeech["test"] = load_dataset("tobiolatunji/afrispeech-200", "isixhosa", split="test", cache_dir=HF_CACHE_DIR)
    print(f"\nAfrispeech-200 {dataset} loaded successfully!\n")

    return afrispeech


def eda(data: DatasetDict, dataset: str) -> None:
    """
    Exploratory data analysis
    """
    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]
    print("Number of training samples: ", len(train_data)) 
    print("Number of validation samples: ", len(val_data)) 
    print("Number of test samples: ", len(test_data)) 
    print()

    # Display the dataset features
    print("Dataset features: ", train_data.features)
    # {'speaker_id': Value(dtype='string', id=None), 
    # 'path': Value(dtype='string', id=None), 
    # 'audio_id': Value(dtype='string', id=None), 
    # 'audio': Audio(sampling_rate=44100, mono=True, decode=True, id=None), 
    # 'transcript': Value(dtype='string', id=None), 
    # 'age_group': Value(dtype='string', id=None), 
    # 'gender': Value(dtype='string', id=None), 
    # 'accent': Value(dtype='string', id=None), 
    # 'domain': Value(dtype='string', id=None), 
    # 'country': Value(dtype='string', id=None), 
    # 'duration': Value(dtype='float32', id=None)}
    print()

    # Count the occurrences of each age group in the dataset
    age_counts_train = pd.Series(train_data["age_group"]).replace("", "UNKNOWN").value_counts(normalize=True)
    age_counts_val = pd.Series(val_data["age_group"]).replace("", "UNKNOWN").value_counts(normalize=True)
    age_counts_test = pd.Series(test_data["age_group"]).replace("", "UNKNOWN").value_counts(normalize=True)
    age_counts_df = pd.DataFrame({
        "train": age_counts_train,
        "val": age_counts_val,
        "test": age_counts_test
    }).fillna(0)
    print("Age group counts (relative frequencies):")
    print(age_counts_df.map(lambda x: f"{x:.2f}"))
    print()

    # Plot distribution
    # Define color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Muted blue, orange, and green
    # plt.style.use("seaborn-whitegrid") 
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)  # High DPI for publication quality
    age_counts_df.plot(
        kind="bar", 
        color=colors, 
        ax=ax,
        width=0.8,  # Adjust bar width for better spacing
        edgecolor="black"  # Make bars more distinguishable
    )
    ax.set_title("Age Group Distribution", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Age Group", fontsize=12, labelpad=8)
    ax.set_ylabel("Relative Frequency", fontsize=12, labelpad=8)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(title="Dataset Split", fontsize=10, title_fontsize=11, loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f"{dataset}_age_group_distribution.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Count the gender distribution in the dataset
    gender_counts_train = pd.Series(train_data["gender"]).replace("", "UNKNOWN").value_counts(normalize=True)
    gender_counts_val = pd.Series(val_data["gender"]).replace("", "UNKNOWN").value_counts(normalize=True)
    gender_counts_test = pd.Series(test_data["gender"]).replace("", "UNKNOWN").value_counts(normalize=True)
    gender_counts_df = pd.DataFrame({
        "train": gender_counts_train,
        "val": gender_counts_val,
        "test": gender_counts_test
    }).fillna(0)
    print("Gender distribution counts (relative frequencies):")
    print(gender_counts_df.map(lambda x: f"{x:.2f}"))
    print()

    # Plot distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)  # High DPI for publication quality
    gender_counts_df.plot(
        kind="bar", 
        color=colors, 
        ax=ax,
        width=0.8,  # Adjust bar width for better spacing
        edgecolor="black"  # Make bars more distinguishable
    )
    ax.set_title("Gender Distribution", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Gender", fontsize=12, labelpad=8)
    ax.set_ylabel("Relative Frequency", fontsize=12, labelpad=8)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(title="Dataset Split", fontsize=10, title_fontsize=11, loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f"{dataset}_gender_distribution.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Count the domain distribution in the dataset
    domain_counts_train = pd.Series(train_data["domain"]).replace("", "UNKNOWN").value_counts(normalize=True)
    domain_counts_val = pd.Series(val_data["domain"]).replace("", "UNKNOWN").value_counts(normalize=True)
    domain_counts_test = pd.Series(test_data["domain"]).replace("", "UNKNOWN").value_counts(normalize=True)
    domain_counts_df = pd.DataFrame({
        "train": domain_counts_train,
        "val": domain_counts_val,
        "test": domain_counts_test
    }).fillna(0)
    print("Domain distribution counts (relative frequencies):")
    print(domain_counts_df.map(lambda x: f"{x:.2f}"))
    print()

    # Plot distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)  # High DPI for publication quality
    domain_counts_df.plot(
        kind="bar", 
        color=colors, 
        ax=ax,
        width=0.8,  # Adjust bar width for better spacing
        edgecolor="black"  # Make bars more distinguishable
    )
    ax.set_title("Domain Distribution", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Domain", fontsize=12, labelpad=8)
    ax.set_ylabel("Relative Frequency", fontsize=12, labelpad=8)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(title="Dataset Split", fontsize=10, title_fontsize=11, loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, f"{dataset}_domain_distribution.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Compute the average duration of the audio files in the dataset
    avg_duration_train = pd.Series(train_data["duration"]).mean()
    avg_duration_val = pd.Series(val_data["duration"]).mean()
    avg_duration_test = pd.Series(test_data["duration"]).mean()
    print("Average audio duration (seconds):")
    print("Train: ", f"{avg_duration_train:.2f}")
    print("Validation: ", f"{avg_duration_val:.2f}")
    print("Test: ", f"{avg_duration_test:.2f}")
    print()

    # Compute the total duration of the audio files in the dataset
    total_duration_train = pd.Series(train_data["duration"]).sum()
    total_duration_val = pd.Series(val_data["duration"]).sum()
    total_duration_test = pd.Series(test_data["duration"]).sum()
    print("Total audio duration (minutes):")
    print("Train: ", f"{total_duration_train / 60:.2f}")
    print("Validation: ", f"{total_duration_val / 60:.2f}")
    print("Test: ", f"{total_duration_test / 60:.2f}")
    print()

    # Display the transcript of the first 3 samples in train, val, test each
    transcripts = {
        "Set": ["Train"] * 3 + ["Validation"] * 3 + ["Test"] * 3,
        "Sample": [1, 2, 3] * 3,
        "Transcript": train_data["transcript"][:3] + val_data["transcript"][:3] + test_data["transcript"][:3]
    }
    transcripts_df = pd.DataFrame(transcripts)
    print("Transcripts of the first 3 samples in train, val, test:")
    print(tabulate(transcripts_df, headers="keys"))
    print()


def prepare_dataset(batch: DatasetDict) -> DatasetDict:
    """
    Helper function that prepares the dataset for training (this function is taken from hugging face tutorial)
    """
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array and pad/truncate to 30 seconds
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR) 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR) 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch


def processing_data(data: DatasetDict, dataset: str) -> DatasetDict:
    """
    Process the data for training:
    - Downsample audio data from 48kHz to 16kHz
    - Compute log-Mel input features from input audio array
    - Encode target text to label ids
    - Save the processed data to disk
    """
    data = data.cast_column("audio", Audio(sampling_rate=16000)) # downsample audio data from 48kHz to 16kHz
    data = data.map(prepare_dataset, remove_columns=data.column_names["train"], num_proc=4)
    data.save_to_disk(f"{PROCESSED_DATA_DIR}_{dataset}")
    print(f"Dataset processed successfully! Saved to {PROCESSED_DATA_DIR}_{dataset}")
    return data


def processing_data_split(data: DatasetDict, dataset: str) -> DatasetDict:
    """
    Process the data for training and split test set by domain.
    """
    # 1. Downsample audio from 48kHz -> 16kHz (same as before)
    data = data.cast_column("audio", Audio(sampling_rate=16000))

    # 2. Map the prepare_dataset function to train/val *as usual* (optional if you want them preprocessed here)
    data["train"] = data["train"].map(prepare_dataset, remove_columns=data["train"].column_names, num_proc=4)
    data["val"]   = data["val"].map(prepare_dataset, remove_columns=data["val"].column_names, num_proc=4)
    
    # 3. CREATE DOMAIN-SPLIT TEST SETS
    test_general  = data["test"].filter(lambda x: x["domain"] == "general")
    test_clinical = data["test"].filter(lambda x: x["domain"] == "clinical")

    # 4. Preprocess those domain splits
    test_general  = test_general.map(prepare_dataset, remove_columns=data["test"].column_names, num_proc=4)
    test_clinical = test_clinical.map(prepare_dataset, remove_columns=data["test"].column_names, num_proc=4)

    # 5. Assemble a new DatasetDict with domain-based test splits
    data_split = DatasetDict({
        "train": data["train"],
        "val":   data["val"],
        "test_general":  test_general,
        "test_clinical": test_clinical
    })

    # 6. SAVE the new domain-split dataset to disk (note the new directory name)
    new_dir = f"{PROCESSED_DATA_DIR}_split_{dataset}"
    data_split.save_to_disk(new_dir)
    print(f"Domain-split dataset saved to {new_dir}")

    return data_split


@app.command()
def main(
    
    dataset: str = typer.Option("isizulu", help="Dataset to load. Must be one of 'isizulu', 'swahili', 'isixhosa'"),
    perform_eda: bool = typer.Option(True),
    process_data: bool = typer.Option(True),
    split_data: bool = typer.Option(False, help="Whether to split the test set by domain (only for isizulu dataset)"
        "This will create a new dataset with two test sets: test_general and test_clinical"),
):
    """
    Main function to process African speech datasets.

    This function loads the specified dataset, optionally performs exploratory data analysis (EDA),
    and processes the data as needed.

    Usage:
        python data_processing.py --dataset [dataset_name] --perform-eda/--no-perform-eda --process-data/--no-process-data
        
    Example:
        python data_processing.py --dataset isizulu --no-perform-eda
        python data_processing.py --help

    Args:
        dataset (str): Dataset to load. Must be one of 'isizulu', 'swahili', 'isixhosa'. Defaults to "isizulu".
        perform_eda (bool): Whether to perform exploratory data analysis. Defaults to True.
        process_data (bool): Whether to process the data. Defaults to True.

    Returns:
        None
    """
    afrispeech = load_data(dataset)
    if perform_eda:
        eda(afrispeech, dataset)
    if process_data:
        if split_data:
            processing_data_split(afrispeech, dataset)
        else:
            processing_data(afrispeech, dataset)

if __name__ == "__main__":
    app()
