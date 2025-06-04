from django.urls import path
from inference.views import *

urlpatterns=[
    path('inference/',Predictor.as_view()), #Get Results
    path('inference/code_llama/',Predictor.as_view()), #Submit Request
]
