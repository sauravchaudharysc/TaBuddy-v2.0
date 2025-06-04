# Import necessary libraries for model training, configuration, and data processing
from trl import DPOTrainer, DPOConfig
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_dataset
from accelerate import Accelerator
import torch
from typing import Dict, Optional
from dataclasses import dataclass, field
import time
import os
import json
import ast
from datetime import datetime

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():

        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


MODEL_DIRECTORY_MAP = {
    "7b" : "/raid/ganesh/nagakalyani/Downloads/CodeLlama-7b-Instruct-hf",
    "22b" : "/raid/ganesh/nagakalyani/nagakalyani/Tushar/codestral"
}

def initialize_model_and_tokenizer_dpo(
    model_size="7b", adapter_path="", device="cuda:0", quantization_config=None
):
    """
        Initialize the model and tokenizer.
    """
    start_time = time.time()

    model_directory_path = MODEL_DIRECTORY_MAP[model_size]
    tokenizer = AutoTokenizer.from_pretrained(
        model_directory_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load the model with quantization if provided
    if quantization_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_directory_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": Accelerator().local_process_index},
        ).eval()
    else:
        # Load the model without quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_directory_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()

    # Loading adapters
    if adapter_path != "":
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    end_time = time.time()
    print(f"Loaded model and tokeniser in {end_time - start_time} seconds")

    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--output_dir", type=str, help="The folder where the fine tuned model is saved"
    )

    parser.add_argument(
        "--train_dataset_path", type=str, help="Path to the train dataset file"
    )
    parser.add_argument(
        "--eval_dataset_path", type=str, help="Path to the eval dataset file"
    )

    args = parser.parse_args()

    torch.manual_seed(0)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer, model = initialize_model_and_tokenizer_dpo(
        model_size=args.model_size, device=args.device, quantization_config=bnb_config
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model.config.use_cache = False

    datset_start_time = time.time()

    train_dataset = load_dataset(
        "json", data_files=args.train_dataset_path, split="train"
    )
    eval_dataset = load_dataset(
        "json", data_files=args.eval_dataset_path, split="train"
    )

    dataset_end_time = time.time()

    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    training_start_time = time.time()

    # Set training configurations (e.g., batch size, evaluation steps)
    training_args = DPOConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=250,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        evaluation_strategy="steps",
        eval_steps=0.5,
        output_dir=args.output_dir,
        report_to="tensorboard",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        optim="paged_adamw_32bit",
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        seed=0,
    )

    # Configure LoRA (Low-Rank Adaptation) parameters for PEFT
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Initialize the DPOTrainer with the above configurations
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,  # Higher beta means lesser diversion from the initial policy
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=1024,
        max_length=1536,
    )

    # Start training
    dpo_trainer.train()

    # Save the trained model with a timestamped directory
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    timestamped_dir = os.path.join(args.output_dir, f"model_{current_time}")

    # Create the directory if it doesn't exist
    os.makedirs(timestamped_dir, exist_ok=True)

    # Save the model inside the timestamped directory
    dpo_trainer.save_model(timestamped_dir)

    print(f"Model saved in: {timestamped_dir}")

    # Save final checkpoint
    checkpoint_dir = os.path.join(timestamped_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(checkpoint_dir)

    training_end_time = time.time()

    # Calculate times
    data_loading_time = dataset_end_time - datset_start_time
    training_time = training_end_time - training_start_time

    # Write times to a file
    time_file_path = os.path.join(checkpoint_dir, "time.txt")
    with open(time_file_path, "w") as time_file:
        time_file.write(f"Data loading time: {data_loading_time:.2f} seconds\n")
        time_file.write(f"Training time: {training_time:.2f} seconds\n")
