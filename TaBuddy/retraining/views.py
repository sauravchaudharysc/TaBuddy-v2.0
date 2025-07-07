from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .services.retrain_service import RetrainService
import subprocess
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

class RetrainAPIView(APIView):
    """
    POST /retrain/       → kick off a new training job
    """
    def post(self, request):
        service = RetrainService(config=request.data)
        return service.submit_task()

    """
    GET  /retrain/?task_id=XXX  → poll its status/result
    """
    def get(self, request):
        task_id = request.query_params.get("task_id")
        service = RetrainService()
        return service.get_status_response(task_id)

@api_view(['GET'])
def get_gpu_details(request):
    """
    GET /gpu-details/ → returns a list of installed GPUs and their memory stats
    """
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,memory.total,memory.free,memory.used',
             '--format=csv,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        lines = result.stdout.strip().splitlines()
        gpu_details = []
        for line in lines:
            gpu_no, total_str, free_str, used_str = [x.strip() for x in line.split(',')]
            
            # parse out the numeric MiB values
            total_val = float(total_str.split()[0])
            used_val  = float(used_str.split()[0])
            
            # compute percent used
            used_pct = round((used_val / total_val) * 100, 2)

            gpu_details.append({
                'gpu_no':       int(gpu_no),
                'total_memory': total_str,
                'free_memory':  free_str,
                'used_memory':  used_str,
                'used_percent': used_pct
            })

        return Response(gpu_details, status=200)

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=500
        )

