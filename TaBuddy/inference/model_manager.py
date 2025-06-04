import torch
import gc
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from django.conf import settings

class ModelManager:
    _model = None
    _tokenizer = None
    _device = None
    _initialized = False

    @classmethod
    def load_model(cls, model_path, adapter_path, device="cuda:0"):
        try :
            torch.cuda.empty_cache()
            gc.collect()

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device).eval()
            if os.path.exists(adapter_path):
                if os.listdir(adapter_path):  # folder is not empty
                    model = PeftModel.from_pretrained(model, adapter_path)
                else:
                    adapter_path="Not Found"
                    pass

            cls._tokenizer = tokenizer
            cls._model = model
            cls._device = device
            cls._initialized = True
            print(f"Loaded model from {model_path} with adapter {adapter_path}")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    @classmethod
    def initialize(cls):
        if not cls._initialized:
            try:
                cls.load_model(settings.MODEL_DIRECTORY_PATH, settings.ADAPTER_PATH)
            except Exception as e:
                print(f"Error during model initialization: {e}")
                sys.exit(1)  # Exit the application

    @classmethod
    def get_model(cls):
        return cls._model

    @classmethod
    def get_tokenizer(cls):
        return cls._tokenizer

    @classmethod
    def reload_model(cls, new_model_path, new_adapter_path):
        cls.load_model(new_model_path, new_adapter_path)
