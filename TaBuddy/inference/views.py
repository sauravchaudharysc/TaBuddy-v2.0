from rest_framework.views import APIView
from rest_framework.response import Response
from celery.result import AsyncResult
from TaBuddy.celery import app as a
from .services.predictor_service import PredictorService
from django.core.cache import cache
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


class Predictor(APIView):
    def post(self, request):
        if 'code_llama' in (request.get_full_path()).lower():
            predictor_obj = PredictorService(data=request.data,files=request.FILES,log_file_name="code_llama_query")
            response = predictor_obj.submit_task()
            return response
        elif 'qwen' in (request.get_full_path()).lower():
            predictor_obj = PredictorService(data=request.data,files=request.FILES,log_file_path="qwen")
            response = predictor_obj.submit_task()
            return response
        else :
            return Response({"error": "Unsupported model specified."}, status=400)
        
    def get(self, request):
        task_id = request.query_params.get('task_id')
        if not task_id:
            return Response({"error": "task_id is required"}, status=404)
        
        # Check if we know this task_id (only for the lifetime of the container)
        if cache.get(f"known_task_{task_id}") is None:
            return Response({"message": "Task Failed"},status=404)
        
        task_result = AsyncResult(task_id)
        print(task_result.status)

        if task_result.status == "SUCCESS":
            return Response(task_result.result, status=200)
        elif task_result.status == "PENDING":
            return Response({"message": "task pending"},status=202)
        else :
            return Response({"message": "task Failed"},status=404)