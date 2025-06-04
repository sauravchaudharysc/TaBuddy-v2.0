from django.urls import path
from . import views

urlpatterns=[
    path('data-point/', views.DataPoint.as_view(), name='data_point')
]
