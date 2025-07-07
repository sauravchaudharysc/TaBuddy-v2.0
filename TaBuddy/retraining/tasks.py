from celery import shared_task
from .services.Utils.utility import *
from django.conf import settings

@shared_task(bind=True)
def retrain_task(self, model_name,model_size,cuda_device):
    """
    config: dict of whatever hyper-params / paths your training needs
    """
    print(f"Retraining task started with model: {model_name}, size: {model_size}, on device: {cuda_device}")
    accuracy = retraining(model_name, model_size, cuda_device)

    return {"accuracy": accuracy}
