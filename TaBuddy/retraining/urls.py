from django.urls import path
from retraining.views import *

urlpatterns = [
   path('retrain/', RetrainAPIView.as_view(), name='retrain'),
   path('gpu-details/', get_gpu_details, name='gpu-details'),
]
