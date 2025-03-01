"""
Download Afrispeech-200 data from huggingface datasets
"""


from datasets import load_dataset, DatasetDict

if __name__ == "__main__":
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
