from datasets import load_from_disk, DatasetDict
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
from peft import LoraConfig, get_peft_model, IA3Config, IA3Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import HF_CACHE_DIR, PROCESSED_DATA_DIR, MODEL_NAME, MODELS_DIR
from data_processing import load_data, processing_data
from datetime import datetime

# ------------------------------------------------------------------------------------
# Helper functions for data collating, training, and evaluation
# ------------------------------------------------------------------------------------

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


class CustomTrainer(Seq2SeqTrainer):
    """
    Custom trainer class to handle IA3 training
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """ Remove any extra keys that the model doesn't support """
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


def make_inputs_require_grad(module, input, output) -> None:
    """
    LoRA helper function.
    Make sure the gradient is propagated all the way to the inputs, to ensure the model weights are trainable
    """
    output.requires_grad_(True)


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


# ------------------------------------------------------------------------------------
# Main library functions
# ------------------------------------------------------------------------------------

def data_pipeline(dataset: str) -> tuple[DatasetDict, WhisperProcessor]:
    """
    Load the training and testing dataset from disk. If the dataset is not found, it will be downloaded and processed.
    Args:
        dataset (str): The name of the dataset to load. Can be either "isixhosa", "isizulu", or "swahili"
    Returns:
        data (DatasetDict): The loaded dataset with train, validation, and test splits
        processor (WhisperProcessor): The Whisper processor for the dataset
    """
    # Load the dataset
    data_dir = f"{PROCESSED_DATA_DIR}_{dataset}"
    if not os.path.exists(data_dir):
        load_data(dataset)
        processing_data(dataset)
    data = load_from_disk(data_dir)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR, language="English", task="transcribe")
    return data, processor


def load_model(peft: str) -> WhisperForConditionalGeneration:
    """
    Load the pre-trained Whisper model and set the generation config
    Args:
        peft (str): One of "partial", "lora", or "ia3". The type of fine-tuning to be used.
    Returns:
        model (WhisperForConditionalGeneration): The pre-trained Whisper model
    """
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language = "English"
    model.generation_config.task = "transcribe"
    if peft == "partial":
        model = setup_partial_finetuning(model)
    elif peft == "lora":
        model = setup_lora(model)
    elif peft == "ia3":
        model = setup_ia3(model)
    return model


def setup_partial_finetuning(model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    """
    Setup the model for partial fine-tuning with LoRA
    Args:
        model (WhisperForConditionalGeneration): The pre-trained Whisper model
    Returns:
        model (WhisperForConditionalGeneration): The model with only the last layer of the decoder and encoder trainable
    """
    # Finetune the last layer of the decoder and the encoder
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.model.decoder.layers[-1].named_parameters():
        if "fc" in name or "final_layer_norm" in name: 
            param.requires_grad = True
    for name, param in model.model.encoder.layers[-1].named_parameters():
        if "fc" in name or "final_layer_norm" in name: 
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    return model


def setup_lora(model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    """
    Setup the model for LoRA fine-tuning
    Args:
        model (WhisperForConditionalGeneration): The pre-trained Whisper model
    Returns:
        model (WhisperForConditionalGeneration): The model with LoRA configuration
    """
    config = LoraConfig(r=4, lora_alpha=64, use_rslora=False, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.model.get_encoder().conv1.register_forward_hook(make_inputs_require_grad)
    model.print_trainable_parameters()
    return model


def setup_ia3(model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    """
    Setup the model for IA3 fine-tuning
    Args:
        model (WhisperForConditionalGeneration): The pre-trained Whisper model
    Returns:
        model (WhisperForConditionalGeneration): The model with IA3 configuration
    """
    config = IA3Config(peft_type="IA3", target_modules=["k_proj", "v_proj", "q_proj", "fc1", "fc2"], feedforward_modules=["fc1", "fc2"])
    model = IA3Model(config=config, model=model, adapter_name="ia3")
    
    # Count and print trainable parameters
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/all_params:.2%} of total)")
    return model


def setup_training_args(peft: str) -> Seq2SeqTrainingArguments:
    """
    Setup the training arguments for the model
    Args:
        peft (str): One of "partial", "lora", or "ia3". The type of fine-tuning to be used.
    Returns:
        training_args (Seq2SeqTrainingArguments): The training arguments for the model
    """
    # PEFT specific training arguments
    max_steps = {"partial": 200, "lora": 100, "ia3": 100}.get(peft, 100)
    learning_rate = {"partial": 1e-3, "lora": 1e-3, "ia3": 5e-4}.get(peft, 1e-3)
    warmup_steps = {"partial": 20, "lora": 0, "ia3": 0}.get(peft, 20)
    scheduler = {"partial": "linear", "lora": "constant", "ia3": "cosine"}.get(peft, "linear")
    remove_unused = {"partial": True, "lora": False, "ia3": False}.get(peft, True)
    label_names = {"partial": None, "lora": ["labels"], "ia3": ["labels"]}.get(peft, ["labels"])
    save_safetensor = {"partial": True, "lora": True, "ia3": False}.get(peft, True)

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir= MODELS_DIR, 
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1, 
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type=scheduler,
        max_steps=max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=100,
        save_steps=100,
        eval_steps=100,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        use_cpu=True,
        use_ipex=False,
        remove_unused_columns=remove_unused,
        label_names=label_names,
        save_safetensors=save_safetensor
    )
    return training_args


def train_model(model: WhisperForConditionalGeneration,
                data: DatasetDict,
                processor: WhisperProcessor,
                peft: str, 
                training_args: Seq2SeqTrainingArguments) -> None:
    """
    Train the model with the given data and processor
    Args:
        model (WhisperForConditionalGeneration): The pre-trained Whisper model
        data (DatasetDict): The loaded dataset with train, validation, and test splits
        processor (WhisperProcessor): The Whisper processor for the dataset
        peft (str): One of "partial", "lora", or "ia3". The type of fine-tuning to be used.
        training_args (Seq2SeqTrainingArguments): The training arguments for the model
    """
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    if peft != "ia3":
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )
    else:
        trainer = CustomTrainer(
        args=training_args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
        
    # Evaluate the pretrained model before fine tuning
    print("Evaluating the pre-trained model...")
    eval_results = trainer.evaluate()
    print("Evaluation results: ", eval_results)

    # Train the model
    print("Training the model...")
    trainer.train()

    # Final evaluation
    print("Evaluating the finetuned model...")
    eval_results = trainer.evaluate()
    print("Evaluation results: ", eval_results)


def main():
    """
    Main function to run the training and evaluation pipeline
    """
    # Load the data and processor
    dataset = "isixhosa"
    data, processor = data_pipeline(dataset)

    # Load the model
    peft = "partial"
    model = load_model(peft)

    # Setup the training arguments
    training_args = setup_training_args(peft)

    # Train the model
    train_model(model, data, processor, peft, training_args)