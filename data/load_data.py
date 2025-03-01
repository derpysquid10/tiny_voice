"""
Download Afrispeech-200 data from huggingface datasets
"""


from datasets import load_dataset

if __name__ == "__main__":
    afrispeech = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="train", cache_dir="data/cache")
    print("Afrispeech-200 dataset loaded successfully!")

    # look at the data
    print("Number of samples: ", len(afrispeech))
    print("First sample metadata: ")
    print(afrispeech[0])
    print("cache file location:")
    print(afrispeech.cache_files)
