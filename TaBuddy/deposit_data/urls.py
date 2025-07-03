from django.urls import path
from .views import DataPoint

urlpatterns = [
    path('data-point/', DataPoint.as_view()),
]
