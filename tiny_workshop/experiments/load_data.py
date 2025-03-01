"""
Download Afrispeech-200 data from huggingface datasets
"""


from datasets import load_dataset, DatasetDict
import os

def set_cache_dir() -> str:
    """
    Set the cache directory for the dataset, tokenizer, and feature extractor
    """
    # Find the root of the git repository to store the cache
    repo_root = os.path.abspath(os.path.dirname(__file__))  # Get current script's directory
    while not os.path.exists(os.path.join(repo_root, ".git")) and repo_root != "/":
        repo_root = os.path.dirname(repo_root)  # Move up one level until we find the .git directory

    cache_dir = os.path.join(repo_root, "data", "cache")
    return repo_root, cache_dir


if __name__ == "__main__":
    repo_root, cache_dir = set_cache_dir()
    afrispeech = DatasetDict()
    afrispeech["train"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="train", cache_dir="data/cache")
    afrispeech["val"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="validation", cache_dir="data/cache")
    afrispeech["test"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="test", cache_dir="data/cache")
    print("Afrispeech-200 dataset loaded successfully!")

    # look at the data
    print("Number of training samples: ", len(afrispeech["train"]))
    print("Number of validation samples: ", len(afrispeech["val"]))
    print("Number of test samples: ", len(afrispeech["test"]))
    print("Dataset info: ", afrispeech)
    print("cache file location:")
    print(afrispeech.cache_files)
