from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import os

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        The data collator is a function that is responsible for taking a list of features and converting them into a torch tensor.
        This is where we do the padding for the model inputs and the labels.
        This is a simplified version of the data collator provided by hugging face.

        Args:
            features: List of features. Each feature is a dictionary containing the input features and the labels.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the model inputs and the labels as correctly padded torch tensors.
        """

        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences and pad labels to max length
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos (beginning of sentence) token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def set_cache_dir() -> str:
    """
    Set the cache directory for the dataset, tokenizer, and feature extractor
    """
    # Find the root of the git repository to store the cache
    repo_root = os.path.abspath(os.path.dirname(__file__))  # Get current script's directory
    while not os.path.exists(os.path.join(repo_root, ".git")) and repo_root != "/":
        repo_root = os.path.dirname(repo_root)  # Move up one level until we find the .git directory

    cache_dir = os.path.join(repo_root, "data/cache")
    return repo_root, cache_dir


def load_data(cach_dir: str) -> DatasetDict:
    """
    Load Afrispeech-200 data from huggingface datasets
    """
    afrispeech = DatasetDict()
    afrispeech["train"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="train", cache_dir=cache_dir)
    afrispeech["val"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="validation", cache_dir=cache_dir)
    afrispeech["test"] = load_dataset("tobiolatunji/afrispeech-200", "isizulu", split="test", cache_dir=cache_dir)
    return afrispeech


def prepare_dataset(batch: DatasetDict) -> DatasetDict:
    """
    Prepare the dataset for training (this function is taken from hugging face tutorial)
    """
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch


def compute_metrics(pred: any, metric=evaluate.load("wer")) -> Dict[str, float]:
    """
    Compute the evaluation metric for the predicted transcript
    """

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



if __name__ == "__main__":
    # Some useful variables
    repo_root, cache_dir = set_cache_dir()
    model_name = "openai/whisper-base"

    # Load the data
    print("Loading Afrispeech-200 dataset...")
    afrispeech = load_data(cache_dir)
    print("Afrispeech-200 dataset loaded successfully!")
    print("Number of training samples: ", len(afrispeech["train"]))
    print("Number of validation samples: ", len(afrispeech["val"]))
    print("Number of test samples: ", len(afrispeech["test"]))
    # Optional: print the features of the dataset
    # print("Dataset features: ", afrispeech["train"].features)

    # Create feature extractor, tokenizer, and processor
    # The feature extractor turns audio into log-Mel spectrogram and automatically truncates/pad to 30 seconds
    # The tokenizer maps words to token ids and adds special tokens like [BOS], [EOS], [PAD], etc.
    # The processor combines the feature extractor and tokenizer for easy access
    print("Loading feature extractor, tokenizer, and processor...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, cache_dir=cache_dir, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir, language="English", task="transcribe")
    print("Feature extractor, tokenizer, and processor loaded successfully!")

    # The audio has a sampling rate of 44.1kHz. Needs to downsample to 16kHz for the model
    print("Downsampling audio to 16kHz...")
    afrispeech = afrispeech.cast_column("audio", Audio(sampling_rate=16000))

    # Prepare the dataset with two columns: input_features (meg spectrogram) and labels (tokenized transcript)
    print("Preparing dataset for training...")
    afrispeech = afrispeech.map(prepare_dataset, remove_columns=afrispeech.column_names["train"], num_proc=4)
    print("Dataset after processing: ")
    print(afrispeech["train"][0].keys())
    print("Dataset prepared successfully!")

    # Load pre-trained model
    print("Loading pre-trained model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.generation_config.language = "English"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Set the data collator, which automatically feeds the data to the model
    print("Setting data collator, eval metrics, and training arguments...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Define the evaluation metric (word error rate)
    wer_metric = evaluate.load("wer")

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{repo_root}/model/baseline_gpu", 
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1, 
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=1000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,
        eval_steps=200,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        use_cpu=False,
    )

    # Create the trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=afrispeech["train"],
        eval_dataset=afrispeech["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Train the model
    print("Training the model...")
    trainer.train()














    



