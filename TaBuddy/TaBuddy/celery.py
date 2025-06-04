import multiprocessing as mp

# Set 'spawn' method for Celery worker processes to avoid CUDA re-initialization errors
mp.set_start_method('spawn', force=True)

import os
from celery import Celery
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TaBuddy.settings')

app = Celery('TaBuddy')
app.conf.enable_utc = False
app.conf.update(timezone='Asia/Kolkata')
app.config_from_object(settings, namespace='CELERY')
app.autodiscover_tasks()

