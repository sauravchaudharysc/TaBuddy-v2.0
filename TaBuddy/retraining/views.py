import sys
import os
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.template import loader
from TaBuddy.settings import BASE_DIR

from retraining_raw.Utils.dpo_train_tushar import run_training

@require_http_methods(["GET", "POST"])
def retrain(request):
    try:
        run_training(
            model_name="CodeLLama-7b",
            device="cuda:0",
            output_dir="./outputs",
            train_dataset_path=f"{BASE_DIR}/retraining_raw/Utils/Dataset/train.jsonl",
            eval_dataset_path=f"{BASE_DIR}/retraining_raw/Utils/Dataset/eval.jsonl",
            test_dataset_path=f"{BASE_DIR}/retraining_raw/Utils/Dataset/test.jsonl"
        )
        return JsonResponse({"message": "Training completed successfully."})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
