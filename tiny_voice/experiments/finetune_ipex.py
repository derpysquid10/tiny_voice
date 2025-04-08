from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import wandb
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import HF_CACHE_DIR, PROCESSED_DATA_DIR, MODEL_NAME, MODELS_DIR

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


def compute_metrics(pred: any) -> Dict[str, float]:
    """
    Compute the evaluation metric for the predicted transcript
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    metric=evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def train_cpu_ipex():
    print("Loading data...")
    afrispeech = load_from_disk(PROCESSED_DATA_DIR)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR, language="English", task="transcribe")

    print("Loading pre-trained model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.generation_config.language = "English"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    print("Setting data collator, eval metrics, and training arguments...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Define the evaluation metric (word error rate)
    wer_metric = evaluate.load("wer")

    wandb.init(
        project="tiny_voice",
        name="finetune_ipex",
        tags=["cpu"],
    )

    # Define the training arguments
    batch_size = 8
    max_steps = 100
    training_args = Seq2SeqTrainingArguments(
        output_dir= MODELS_DIR / "ipex", 
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1, 
        learning_rate=1e-5,
        warmup_steps=20,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=100,
        save_steps=25,
        eval_steps=25,
        logging_steps=5,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        use_cpu=True,
        use_ipex=True,
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

    # Evaluate the pretrained model before fine tuning
    # print("Evaluating the pre-trained model...")
    # eval_results = trainer.evaluate()
    # print("Evaluation results: ", eval_results)

    # Train the model
    print("Training the model...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    time_per_sample = (end_time - start_time) / (max_steps * batch_size)
    wandb.log({"time_per_sample": time_per_sample})

if __name__ == "__main__":
    train_cpu_ipex()