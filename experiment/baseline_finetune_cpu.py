from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer

import os


def set_cache_dir() -> str:
    """
    Set the cache directory for the dataset, tokenizer, and feature extractor
    """
    # Find the root of the git repository to store the cache
    repo_root = os.path.abspath(os.path.dirname(__file__))  # Get current script's directory
    while not os.path.exists(os.path.join(repo_root, ".git")) and repo_root != "/":
        repo_root = os.path.dirname(repo_root)  # Move up one level until we find the .git directory

    cache_dir = os.path.join(repo_root, "data/cache")
    return cache_dir


def load_data(cach_dir: str) -> DatasetDict:
    """
    Load Afrispeech-200 data from huggingface datasets
    """
    afrispeech = DatasetDict()
    afrispeech["train"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="train", cache_dir=cache_dir)
    afrispeech["val"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="validation", cache_dir=cache_dir)
    afrispeech["test"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="test", cache_dir=cache_dir)
    print("Afrispeech-200 dataset loaded successfully!")
    print("Number of training samples: ", len(afrispeech["train"]))
    print("Number of validation samples: ", len(afrispeech["val"]))
    print("Number of test samples: ", len(afrispeech["test"]))
    # Optional: print the features of the dataset
    # print("Dataset features: ", afrispeech["train"].features)
    return afrispeech

def prepare_dataset(batch: DatasetDict) -> DatasetDict:
    """
    Prepare the dataset for training (this function is taken from hugging face)
    """
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch


if __name__ == "__main__":
    # Some useful variables
    cache_dir = set_cache_dir()
    model_name = "openai/whisper-base"

    # Load the data
    afrispeech = load_data(cache_dir)

    # Create feature extractor, tokenizer, and processor (which combines extractor and tokenizer, useful for later)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, cache_dir=cache_dir, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir, language="English", task="transcribe")

    # The audio has a sampling rate of 44.1kHz. Needs to downsample to 16kHz for the model
    afrispeech = afrispeech.cast_column("audio", Audio(sampling_rate=16000))

    # Prepare the dataset with two columns: input_features (meg spectrogram) and labels (tokenized transcript)
    afrispeech = afrispeech.map(prepare_dataset, remove_columns=afrispeech.column_names["train"], num_proc=4)

    print("Dataset after processing: ")
    print(afrispeech["train"][0].keys())






    



