import gc
import torch
import warnings
import os
import atexit
import signal
import sys
from django.apps import AppConfig
from django.conf import settings
from .model_manager import ModelManager  # Import the ModelManager class

# Suppress transformer warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Cleanup function to be called on exit or signal
def cleanup(signum=None, frame=None):
    torch.cuda.empty_cache()
    print("Clearing GPU cache")
    if signum is not None:
        sys.exit(0)

class InferenceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'inference'

    # Ensure the model and tokenizer are loaded when the app starts
    def ready(self):
        # Skip initialization during management commands like migrate/makemigrations or Flower
        skip_commands = ['makemigrations', 'migrate', 'shell', 'createsuperuser', 'flower', 'collectstatic']
        if any(cmd in sys.argv for cmd in skip_commands):
            return

        # Prevent reloading in every worker in production
        if not os.environ.get('RUN_MAIN'):
            print("Inference App: Initializing model...")
            ModelManager.initialize()

    @classmethod
    def reload_model(cls, new_model_path, new_adapter_path):
        """
        Method to reload the model and adapter at runtime.
        """
        ModelManager.reload_model(new_model_path, new_adapter_path)

# Register cleanup function for application shutdown
atexit.register(cleanup)
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)
