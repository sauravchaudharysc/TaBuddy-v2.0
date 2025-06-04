from djongo import models
from django.contrib.auth.models import AbstractUser
from bson import ObjectId

class CustomUser(AbstractUser):
    _id = models.ObjectIdField(primary_key=True, default=ObjectId, editable=False)

    # Optional: override default `id` property for compatibility
    @property
    def id(self):
        return str(self._id)

