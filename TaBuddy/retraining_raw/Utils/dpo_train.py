# Import necessary libraries for model training, configuration, and data processing
from trl import DPOTrainer, DPOConfig
from transformers import GenerationConfig
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from accelerate import Accelerator
import torch
from datetime import datetime
import os
import time
import mlflow
import json
import random
import time
import torch
import ast
import os
from django.conf import settings

        
def json_from_string (string) : 
    return ast.literal_eval(string.strip())           

def extract_rating(model_response):
    stripped_model_response = model_response.strip()
    start_index = model_response.find('{')
    end_index = model_response.find('}') + 1

    content_within_braces = model_response[start_index:end_index]
    already_extracted = 1
    try : 
        
        extracted_ans = json_from_string(content_within_braces)
        already_extracted = 0
    except :
        if (stripped_model_response.startswith('''{\n"answer": "''')) :
            option = stripped_model_response[13]
        elif (stripped_model_response.startswith('''{\"answer\" : ''')) :
            option = stripped_model_response[12] 
        elif (stripped_model_response.startswith("The correct answer is ")) : 
            option = stripped_model_response[22]
        elif (stripped_model_response.startswith("Answer: ")):
            option = stripped_model_response[8]
        else : 
            return 0
    reasoning="I am unable to provide the reasoning for this criterion."        
    if not (already_extracted) : 
        try:
            option = extracted_ans['answer'][0]
        except Exception as e:
            return 0

    try : 
        option = option.capitalize()
    except Exception as e : 
        pass
        
    diff = ord(option) - ord('A')
    if not(diff >= 0 and diff < 4) : 
        # print(student_id, model_response[:20])
        return 0

    return option


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
    "CodeLLama-7b": "/raid/ganesh/nagakalyani/Downloads/CodeLlama-7b-Instruct-hf",
    "CodeStral-22b": "/raid/ganesh/nagakalyani/nagakalyani/Tushar/codestral"
}

def initialize_model_and_tokenizer_dpo(model="CodeLLama-7b", adapter_path="", device="cuda:0", quantization_config=None):
    """
    Initialize the model and tokenizer.
    """
    start_time = time.time()
    model_directory_path = MODEL_DIRECTORY_MAP[model]

    tokenizer = AutoTokenizer.from_pretrained(model_directory_path, trust_remote_code=True)
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
    print(f"Loaded model and tokenizer in {end_time - start_time:.2f} seconds")
    return tokenizer, model

def generate_single_response(model, tokenizer, prompt, max_length=1024, device="cuda:0"):
    """
    Generates a response for a single user prompt.

    Args : 
        model : The model which has been loaded into memory
        tokenizer : The tokenizer which has been loaded into memory
        max_length (int) : The maximum input length
        prompt (str) : The prompt to be passed to the model
        device (str) : The device on which the inference is going to run 

    Returns : 
        A string response from the model
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length ,add_special_tokens=False).to(device)

    output = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.0,
        max_new_tokens=512,
        output_scores=True
    )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="CodeLLama-7b")
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
    parser.add_argument(
        "--test_dataset_path", type=str, help="Path to the test dataset file"
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
        model=args.model,
        device=args.device,
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    datset_start_time = time.time()
    train_dataset = load_dataset("json", data_files=args.train_dataset_path, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset_path, split="train")
    dataset_end_time = time.time()
    
    
    # Training configuration
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

    # LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set tracking URI and experiment name
    mlflow.set_tracking_uri("http://10.195.100.5:5000")
    mlflow.set_experiment("DPO_Training_Experiments")
    
    run_name = f"DPO_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # Log configuration parameters
        mlflow.log_param("model", args.model)
        mlflow.log_param("lora_r", args.lora_r)
        mlflow.log_param("lora_alpha", args.lora_alpha)
        mlflow.log_param("lora_dropout", args.lora_dropout)
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("epochs", training_args.num_train_epochs)
        
        # Trainer initialization
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            beta=0.1,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_prompt_length=1024,
            max_length=1536,
        )

        # Training
        training_start_time = time.time()
        dpo_trainer.train()
        training_end_time = time.time()

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

        # Log times
        data_loading_time = dataset_end_time - datset_start_time
        training_time = training_end_time - training_start_time
        mlflow.log_metric("data_loading_time_sec", data_loading_time)
        mlflow.log_metric("training_time_sec", training_time)

        # Save time.txt
        time_file_path = os.path.join(checkpoint_dir, "time.txt")
        with open(time_file_path, "w") as time_file:
            time_file.write(f"Data loading time: {data_loading_time:.2f} seconds\n")
            time_file.write(f"Training time: {training_time:.2f} seconds\n")

        # Log full model dir as artifact
        mlflow.log_artifacts(timestamped_dir, artifact_path="model")

        # Accuracy Calculation 
        tokenizer,model = initialize_model_and_tokenizer_dpo(
            model=args.model,
            device=args.device,
            adapter_path=checkpoint_dir,
            quantization_config=bnb_config
        )

        # Load the test dataset
        test_data = {}
        id=0
        # Read the .jsonl file line by line 
        with open(args.test_dataset_path, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                # Parse each line as a JSON object
                line_data = json.loads(line.strip())
                test_data[id]= line_data
                id+=1
        prompts = []
        for i in range(len(test_data)):
            prompts.append(test_data[i]['prompt'])
        responses = []
        for i in range(len(test_data)):
            responses.append(test_data[i]['chosen'])

        no_of_not_extracted_response = 0
        total = 0 
        truepredicted = 0 
        for i in range(len(prompts)):
            model_response = generate_single_response(
                        model,
                        tokenizer,
                        prompts[i],
                        1024,
                        args.device
                    )

            llmrating = extract_rating(model_response)
            testrating = extract_rating(responses[i])
            if not llmrating or not testrating:
                no_of_not_extracted_response += 1
                continue
            total += 1
            if llmrating.strip() == testrating.strip() :
                truepredicted+=1
        accuracy = 1
        mlflow.log_metric("accuracy", accuracy)
        
            
