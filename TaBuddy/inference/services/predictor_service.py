import os
import json
import requests
from datetime import datetime
from django.core.cache import cache
from django.conf import settings
from rest_framework.response import Response

from ..tasks import query_codellama


class PredictorService:

    def __init__(self, data = None,files=None, log_file_name='general'):
        self.data = data
        self.files = files
        self.log_file_path = os.path.join(settings.LOG_DIR,'Inference', f'{log_file_name}.txt')

    def validate_criteria(self,criteria):
        """
        Validate the structure and rating titles of rubric criteria.
        Returns a list of failed criteria.
        """
        def check_criterion_rating(rating_titles):
            ch = 'A'
            for rating in rating_titles:
                if ch != rating:
                    return False
                ch = chr(ord(ch) + 1)
            return True

        def extract_rating(rating_list):
            rating_titles = []
            for rating in rating_list:
                rating_titles.append(rating['title'])
            return sorted(rating_titles)
        
        failed_criteria = []
        for criterion in criteria:
            rating_titles = extract_rating(criterion['Ratings'])
            if not check_criterion_rating(rating_titles):
                failed_criteria.append(criterion['title'])
        return failed_criteria

    def log_submission(self,problem_statement,criteria,submissions):
        """
        Create a timestamped log entry with the problem statement, criteria, and submission IDs.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        submission_ids = list(submissions.keys())
        criteria_pretty = json.dumps(criteria, indent=4)

        log_entry = f"""
        ==================== LOG ENTRY ====================
        Timestamp       : {timestamp}

        Problem Statement:
        {problem_statement}

        Criteria:
        {criteria_pretty}

        Submission IDs  : {submission_ids}
        ===================================================
        """

        with open(self.log_file_path, "a") as log_file:
            log_file.write(log_entry)

    def submit_task(self):
        """
        Submit the grading task asynchronously to Celery.
        Returns the task ID.
        """
        
        problem_statement = self.data.get('problem_statement')
        criteria = json.loads(self.data.get('criteria'))
        submissions = {}
        for key, file_list in self.files.lists():
            file_content = file_list[0].read().strip()
            submissions[key] = file_content
        
        # Validate criteria
        failed_criteria = self.validate_criteria(criteria)
        if len(failed_criteria):
            return Response(
                status=400,
                data={'message': f'Rubric Validation Failed', 'criteria':failed_criteria}
            )

        # Log the submission
        self.log_submission(problem_statement,criteria,submissions)

        try:        
            task = query_codellama.apply_async(
                args=[submissions, problem_statement, criteria],
                kwargs={
                    'criterion_name': "",
                    'max_length': 4096,
                    'few_shot': False,
                    'few_shot_examples': 0,
                    'train_split': 0.7
                }
            )
            cache.set(f"known_task_{task.id}", True, timeout=None)
            # return task.id
            return Response({'status code': 202, 'task': task.id}, status=202)
        except Exception as e:
            print("Exception occurred:", e)
            return Response(status=404, data={'status': 404, 'message': 'Something went wrong'})



