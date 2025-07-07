from celery.result import AsyncResult
from django.core.cache import cache
from TaBuddy.celery import app as celery_app
from ..tasks import retrain_task
from rest_framework.response import Response
from rest_framework import status

class RetrainService:
    def __init__(self, config=None):
        config = config or {}
        self.model_name  = config.get('model_name')
        self.model_size  = config.get('model_size')
        self.cuda_device = config.get('cuda_device')

    def submit_task(self):
        missing = [k for k in ('model_name','model_size','cuda_device')
                   if getattr(self, k) is None]
        if missing:
            return Response(
                {"error": f"Missing parameters: {', '.join(missing)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        # 1) enqueue
        async_result = retrain_task.delay(
            self.model_name,
            self.model_size,
            self.cuda_device
        )

        # 2) remember so GET can validate
        cache.set(f"known_task_{async_result.id}", True, None)

        # 3) return DRF Response
        return Response(
            {"task_id": async_result.id},
            status=status.HTTP_202_ACCEPTED
        )

    def get_status_response(self, task_id):
        # missing ?
        if not task_id:
            return Response(
                {"error": "task_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # unknown or expired ?
        if not cache.get(f"known_task_{task_id}"):
            return Response(
                {"message": "Unknown or expired task_id"},
                status=status.HTTP_404_NOT_FOUND
            )

        # inspect Celery
        result = AsyncResult(task_id, app=celery_app)
        if result.status == "PENDING":
            return Response(
                {"message": "task pending"},
                status=status.HTTP_202_ACCEPTED
            )
        elif result.status == "SUCCESS":
            return Response(
                result.result,
                status=status.HTTP_200_OK
            )
        else:
            return Response(
                {"message": "task failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

