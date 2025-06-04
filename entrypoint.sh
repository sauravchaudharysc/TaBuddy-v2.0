#!/bin/bash
set -e  # exit on error

# Run DB migrations
python manage.py makemigrations
python manage.py migrate

# Create default superuser
python manage.py shell -c "from django.contrib.auth import get_user_model; \
User = get_user_model(); \
User.objects.filter(username='admin').exists() or \
User.objects.create_superuser('admin', 'admin@example.com', 'admin')"

# Collect static files
python manage.py collectstatic --noinput

# Start the Django server
exec python manage.py runserver 0.0.0.0:8000
