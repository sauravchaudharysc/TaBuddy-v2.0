from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services.datapoint_service import DataPointService

class DataPoint(APIView):
    def post(self, request):
        try:
            processor = DataPointService(data=request.data,log_file_name="data-deposit")
            result = processor.submit_task()
            return result
        except Exception as e:
            return Response({"error": str(e)}, status=400)

    def get(self, request):
        return Response({"message": "GET request received"}, status=status.HTTP_200_OK)
