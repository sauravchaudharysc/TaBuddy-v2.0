from rest_framework import response

class GpuDetailsService:
    def get_gpu_details(self):
        #use nvidia-smi
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader'], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise Exception("Error executing nvidia-smi: " + result.stderr.strip())
            gpu_info = result.stdout.strip().split('\n')
            gpu_details = []
            for info in gpu_info:
                name, total_memory, free_memory, used_memory = info.split(', ')
                gpu_details.append({
                    'name': name,
                    'total_memory': total_memory,
                    'free_memory': free_memory,
                    'used_memory': used_memory
                })
            return response.Response(gpu_details, status=200)
        except Exception as e:
            return response.Response({"error": str(e)}, status=500)