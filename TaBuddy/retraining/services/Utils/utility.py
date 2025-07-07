import os
import time
import json
import torch
import mlflow
from datetime import datetime
from trl import DPOTrainer, DPOConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from django.conf import settings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .training_utils import (
    initialize_model_and_tokenizer_dpo,
    generate_single_response,
    extract_rating,
    evaluate_classification
)


def retraining(model_name: str, model_size: str, cuda_device: str) -> float:
    """
    Full retraining pipeline. Returns the final classification accuracy as float.
    """
    # 1. Setup
    torch.manual_seed(0)
    mlflow.set_tracking_uri(settings.MLFLOW_URI)
    mlflow.set_experiment("DPO_Training_Experiment")
    run_name = f"DPO_Run_{datetime.now():%Y%m%d_%H%M%S}"

    # 2. Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # 3. Initialize
    tokenizer, model = initialize_model_and_tokenizer_dpo(
        model=model_name,
        device=cuda_device,
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 4. Load datasets
    train_ds = load_dataset("json", data_files="retraining/services/Utils/Dataset/train.jsonl", split="train")
    eval_ds  = load_dataset("json", data_files="retraining/services/Utils/Dataset/eval.jsonl",  split="train")
    test_path = "retraining/services/Utils/Dataset/test.jsonl"

    # 5. MLflow run
    with mlflow.start_run(run_name=run_name):
        # log hyperparameters
        mlflow.log_params({
            "model_name": model_name,
            "model_size": model_size,
            "device": cuda_device,
            "learning_rate": 1e-5,
            "batch_size": 1,
            "epochs": 1
        })

        # 6. Trainer
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=DPOConfig(
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
                output_dir=settings.OUTPUT_DIR,
                report_to="tensorboard",
                lr_scheduler_type="cosine",
                warmup_steps=100,
                optim="paged_adamw_32bit",
                bf16=True,
                remove_unused_columns=False,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                seed=0,
            ),
            beta=0.1,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            peft_config=LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["q_proj","v_proj","k_proj","out_proj","fc_in","fc_out","wte"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
            max_prompt_length=1024,
            max_length=1536,
        )
        dpo_trainer.train()

        # 7. Save model
        timestamped_dir = os.path.join(settings.OUTPUT_DIR, f"model_{datetime.now():%Y%m%d_%H%M%S}")
        checkpoint_dir = os.path.join(timestamped_dir, "final_checkpoint")
        os.makedirs(timestamped_dir, exist_ok=True)
        dpo_trainer.model.save_pretrained(checkpoint_dir)
        mlflow.log_artifacts(timestamped_dir, artifact_path="model")

        # 8. Evaluate on test set
        tokenizer,model = initialize_model_and_tokenizer_dpo(
            model=model_name,
            device=cuda_device,
            adapter_path=checkpoint_dir,
            quantization_config=bnb_config
        )
        prompts, answers = [], []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                prompts.append(obj['prompt'])
                answers.append(obj['chosen'])

        y_true, y_pred = [], []
        for p, a in zip(prompts, answers):
            resp = generate_single_response(model, tokenizer, p, max_length=1024, device=cuda_device)
            r_pred = extract_rating(resp)
            r_true = extract_rating(a)
            if r_pred and r_true:
                y_pred.append(r_pred)
                y_true.append(r_true)
        
        metrics = evaluate_classification(y_true, y_pred)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        return metrics['accuracy']
