import time
import ast
from accelerate import Accelerator
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from django.conf import settings
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Map for local model directories
MODEL_DIRECTORY_MAP = {
    "CodeLLama-7b": settings.CODELLAMA_PATH,
    "CodeStral-22b": settings.CODESTRAL_PATH
}

def json_from_string(string):
    return ast.literal_eval(string.strip())

def extract_rating(model_response):
    stripped = model_response.strip()
    # attempt JSON-style extraction
    start = model_response.find('{')
    end = model_response.find('}') + 1
    content = model_response[start:end]
    parsed = None
    try:
        parsed = json_from_string(content)
        answer = parsed.get('answer', '')
        option = answer[0] if answer else None
    except Exception:
        # fallback patterns
        option = None
        if stripped.startswith('{\n"answer": "'):
            option = stripped[13]
        elif stripped.startswith('{"answer" : '):
            option = stripped[12]
        elif stripped.startswith('The correct answer is '):
            option = stripped[22]
        elif stripped.startswith('Answer: '):
            option = stripped[8]
    if not option:
        return 0
    option = option.capitalize()
    diff = ord(option) - ord('A')
    if not (0 <= diff < 4):
        return 0
    return option


def initialize_model_and_tokenizer_dpo(model="CodeLLama-7b", adapter_path="", device="cuda:0", quantization_config=None):
    """
    Initialize the model and tokenizer for DPO training.
    """
    start = time.time()
    model_dir = MODEL_DIRECTORY_MAP[model]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    if quantization_config:
        # load with quantization
        model_ckpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
            # device_map={"": device},
            device_map="cuda:0",
        ).eval()
    else:
        # load full precision
        model_ckpt = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()

    if adapter_path:
        model_ckpt = PeftModel.from_pretrained(model_ckpt, adapter_path)
        model_ckpt.eval()

    elapsed = time.time() - start
    print(f"Loaded model and tokenizer in {elapsed:.2f} seconds")
    return tokenizer, model_ckpt


def generate_single_response(model, tokenizer, prompt, max_length=1024, device="cuda:0"):
    """
    Generate a single completion for the given prompt.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    ).to("cuda:0")

    # Convert input tensors to bfloat16 if model expects that
    for key in inputs:
        if inputs[key].dtype == torch.float:
            inputs[key] = inputs[key].to(torch.bfloat16)
            
    outputs = model.generate(
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
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_classification(y_true, y_pred):
    """
    Compute confusion matrix and macro metrics for classification.
    """
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "labels": labels,
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
