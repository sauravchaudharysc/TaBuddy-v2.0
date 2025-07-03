from ..models import Problem, Submission, Criteria, Rating, GradingHistory
from django.utils import timezone
from django.db import transaction
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os
from datetime import datetime


class DataPointService:

    def __init__(self, data, log_file_name='data_deposit'):
        self.data = data
        self.user_id = data.get('user_id')
        self.lab_data = data.get('0', {})
        self.log_file_path = os.path.join(settings.LOG_DIR,'Data-Deposit', f'{log_file_name}.txt')
    
    def dummy(self, student_id):
        return student_id[::-1]

    def submit_task(self):
        try:
            problems_already_present = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                "\n"
                "==================== LOG ENTRY ====================\n"
                f"Timestamp       : {timestamp}\n"
                f"User ID         : {self.user_id}\n"
                "===================================================\n"
            )

            with open(self.log_file_path, "a") as log_file:
                log_file.write(log_entry)
                for lab_name in self.lab_data:
                    lab_id = int(lab_name)

                    for program_id_str, program_data in self.lab_data[lab_name].items():
                        program_id = int(program_id_str)
                        problem_statement = program_data['problem_statement']
                        filename = program_data['file_name']

                        problem, created = Problem.objects.get_or_create(
                            id=program_id,
                            defaults={
                                'problem_statement': problem_statement,
                                'user_id': self.user_id,
                                'lab_id': lab_id
                            }
                        )
                        if not created:
                            log_file.write(f"Problem with ID {program_id} already exists\n")
                        else:
                            log_file.write("Problem object created successfully\n")

                        criteria_list = program_data['rubric']
                        for criterion in criteria_list:
                            for criterion_id, details in criterion.items():
                                title = details['title']
                                description = details['description']

                                criteria_obj, created = Criteria.objects.get_or_create(
                                    id=int(criterion_id),
                                    problem=problem,
                                    defaults={
                                        'title': title,
                                        'description': description
                                    }
                                )
                                if not created:
                                    log_file.write(f"Criteria with ID {criterion_id} already exists\n")
                                else:
                                    log_file.write("Criteria object created successfully\n")

                                for rating_id, rating_data in details['ratings'].items():
                                    try:
                                        Rating.objects.get_or_create(
                                            id=int(rating_id),
                                            criteria=criteria_obj,
                                            defaults={
                                                'title': rating_data['title'],
                                                'description': rating_data['description'],
                                                'marks': int(rating_data.get('marks', 0))
                                            }
                                        )
                                        if not created:
                                                log_file.write(f"Rating with ID {rating_id} already exists\n")
                                        else:
                                            log_file.write("Rating object created successfully\n")
                                    except Exception as e:
                                        log_file.write(f"Error while creating rating: {str(e)}\n")

                        for student_id, submission_data in program_data['student_submissions'].items():
                            dummy_id = self.dummy(student_id)
                            source_code = submission_data['source_code']

                            submission, created = Submission.objects.get_or_create(
                                filename=filename,
                                student_id=student_id,
                                dummy_id=dummy_id,
                                source_code=source_code,
                                problem=problem
                            )

                            if not created:
                                log_file.write(f"Submission with ID {submission.id} already exists\n")
                            else:
                                log_file.write("Submission object created successfully\n")

                            for criterion_id, manual_data in submission_data['manual_rating'].items():
                                criteria_obj = Criteria.objects.filter(problem=program_id, id=int(criterion_id)).first()
                                if not criteria_obj:
                                    continue

                                manual_rating_id, manual_comments = manual_data
                                manual_rating = Rating.objects.filter(id=int(manual_rating_id)).first() if manual_rating_id not in [None, 'None'] else None

                                ai_rating_id, ai_comments = submission_data['ai_rating'].get(criterion_id, (None, ''))
                                ai_rating = Rating.objects.filter(id=int(ai_rating_id)).first() if ai_rating_id not in [None, 'None'] else None

                                existing = GradingHistory.objects.filter(
                                    submission=submission,
                                    criteria=criteria_obj
                                ).exists()

                                if not existing:
                                    GradingHistory.objects.create(
                                        submission=submission,
                                        criteria=criteria_obj,
                                        manual_rating=manual_rating,
                                        ai_rating=ai_rating,
                                        manual_comments=manual_comments,
                                        ai_comments=ai_comments
                                    )
                                else:
                                    log_file.write(f"Entry already exists for program_id={program_id}, student_id={student_id}, criterion_id={criterion_id}\n")


            return Response({"message": "Data received"}, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Error: {str(e)}")
            return Response({"error": "An error occurred"}, status=status.HTTP_400_BAD_REQUEST)
