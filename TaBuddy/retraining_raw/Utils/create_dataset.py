import csv
import argparse
import random
import os
import json

def format_user_prompt (prompt, system_prompt=""):
    """
    Formats a single input string to a CodeLlama compatible format.

    Args : 
        prompt (str) : The user prompt
        system_prompt (str) : The system prompt (Optional)

    Returns : 
        A prompt format compatible with CodeLlama
    """
    if (system_prompt) :
        formatted_prompt = f"<s>[INST] <<SYS>>\\n{system_prompt}\\n<</SYS>>\\n\\n{prompt} [/INST]"
    else : 
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    return formatted_prompt

def get_submission_json(course_code):
    '''
    Returns all the student submissions for a given problem statement

    Args :
        course_code (str) : The path to the submissions for that lab

    Returns : 
        submissions_data (dict) : A dictionary of student submissions. Student IDs are the keys
    '''
    submissions_dir = os.path.join(course_code, "submissions")
    submissions_data = {}

    # Traverse through each submission directory
    for submission_dir in os.listdir(submissions_dir):
        submission_path = os.path.join(submissions_dir, submission_dir)
        if os.path.isdir(submission_path):
            submission_files = os.listdir(submission_path)
            for file_name in submission_files:
                # Assuming all submissions are C++ files
                if file_name.endswith(".cpp"):
                    file_path = os.path.join(submission_path, file_name)
                    with open(file_path, 'r') as file:
                        submission_code = file.read()
                    submission_key = submission_dir.split('@')[0]
                    submissions_data[submission_key] = submission_code.strip()
                    break  # Assuming there's only one .cpp file per submission

    return submissions_data

def get_rubrics(course_code):
    '''
    Fetches all the rubrics and corresponding descriptions associated with a problem statement

    Args : 
        course_code (str) : The path to the submissions for the problem statement

    Returns : 
        parsed_rubrics (dict) : A dictionary of rubric titles, descriptions and ratings.
    '''
    json_file = os.path.join(course_code, "rubrics.json")
    # json_file = os.path.join(course_code, "rubrics1.json")
    with open(json_file, 'r') as f:
        rubrics_data = json.load(f)

    parsed_rubrics = {}
    for item in rubrics_data:
        title = item['title']
        description = item['description']
        ratings = {rating['title']: rating['description']
                   for rating in item['Ratings']}
        parsed_rubrics[title] = {
            'description': description, 'ratings': ratings}

    return parsed_rubrics

def extract_all_original_grades(grades_file_path):
    '''
    A dictionary with criteria as keys. The value for each criterion will be another dictionary with student ids as keys and grades as values
    '''
    with open(grades_file_path, "r") as f:
        reader = csv.reader(f)

        # All the rows in the csv
        rows = []
        for row in reader:
            rows.append(row)

        # Start and end indices for each criterion
        criterion_indices = {}

        current_criterion = ""
        for i in range(1, len(rows[0])):
            if (not current_criterion):
                current_criterion = rows[0][i]
                start_idx = i
            elif (current_criterion != rows[0][i]):
                end_idx = i - 1
                criterion_indices[current_criterion] = [start_idx, end_idx]
                current_criterion = rows[0][i]
                start_idx = i

        criterion_indices[current_criterion] = [start_idx, len(rows[0]) - 1]

        criterion_rating_titles = {}
        for criterion in criterion_indices.keys():
            start_idx = criterion_indices[criterion][0]
            end_idx = criterion_indices[criterion][1]

            rating_titles = []
            for idx in range(start_idx, end_idx + 1):
                rating_titles.append(rows[2][idx])

            criterion_rating_titles[criterion] = rating_titles

        grades = {}
        for criterion in criterion_indices.keys():
            grades[criterion] = {}
            start_idx = criterion_indices[criterion][0]
            end_idx = criterion_indices[criterion][1]

            for i in range(6, len(rows)):
                student_id = rows[i][0]
                student_id = student_id.split('@')[0]

                for idx in range(start_idx, end_idx + 1):
                    if (rows[i][idx] == '1'):
                        grades[criterion][student_id] = criterion_rating_titles[criterion][idx - start_idx]

        return grades

def process_reasoning(lab_path):
    grades_file_path = None

    # Check if the directory exists and find the CSV file
    if os.path.isdir(lab_path):
        for file in os.listdir(lab_path):
            if file == "rubric_ratings.csv":
                grades_file_path = os.path.join(lab_path, file)
                break

    if not grades_file_path:
        raise FileNotFoundError(
            "rubric_ratings.csv not found in the provided lab_path")

    # Process the CSV file
    with open(grades_file_path, "r") as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    crit_ratings = {}
    for i in range(len(rows[0])):
        if i == 0:
            continue
        if rows[0][i] not in crit_ratings:
            crit_ratings[rows[0][i]] = {}
            crit_ratings[rows[0][i]][rows[2][i]] = rows[3][i]
        else:
            crit_ratings[rows[0][i]][rows[2][i]] = rows[3][i]

    student_reasoning2 = {}
    for j in range(len(rows)):
        if j <= 5:
            continue
        id = rows[j][0].split('@')[0]
        for i in range(len(rows[0])):
            if i == 0:
                continue
            if rows[0][i] not in student_reasoning2:
                student_reasoning2[rows[0][i]] = {}
            if id not in student_reasoning2[rows[0][i]]:
                student_reasoning2[rows[0][i]][id] = {}
            if rows[j][i] == '0' or rows[j][i] == 'No Comments' or rows[j][i] == '1':
                student_reasoning2[rows[0][i]][id][rows[2]
                                                   [i]] = crit_ratings[rows[0][i]][rows[2][i]]
            else:
                student_reasoning2[rows[0][i]][id][rows[2][i]] = rows[j][i]

    return student_reasoning2


def create_dpo_prompt(context, code, task, options):
    """
    Creates prompt for finetuning with DPO

    Args : 
        context (str) : The simplified problem statement
        code (str) : The student code
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
    Returns : 
        prompt (str) : A prompt for finetuning with DPO
    """
    options_list = ""
    for key in sorted(options.keys()):
        options_list += f"{key}. {options[key]}\n"

    prompt = '''### Context : 
{}

### Code : 
{}

### Task :
{}

### Options :
{}
### Response : The required output in json format is :'''.format(context, code, task, options_list)
    return prompt   # New format in the last line


def get_dpo_dataset(context, codes, task, options, original_grades, system_prompt, split=0.7):
    """
    Creates dataset for a single lab for finetuning with DPO

    Args : 
        context (str) : The simplified problem statement
        codes (dict) : A dictionary of student codes. Student ids are the keys
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
        original_grades (dict) : A dictionary of TA assigned grades. Student ids are the keys
        system_prompt (str) : The system prompt 
        split (float) : The train split for the dataset
    Returns : 
        A tuple of train and test splits. Both the splits are lists
        Each item in the split is a json with three fields "prompt", "chosen" and "rejected"
    """
    lora_prompts = {}
    chosen = {}
    rejected = {}

    for student_id in sorted(codes.keys()):
        if student_id not in original_grades:
            continue
        original_grade = original_grades[student_id]
        # chosen_response = f'The correct answer is {original_grade}. {options[original_grade]} </s>'
        original_grade = original_grade.strip()
        chosen_response = '''{"answer" : ''' + \
            f'''"{original_grade}. {options[original_grade]}"''' + '''} </s>'''

        rejected_responses = []
        for option in options.keys():
            if (option != original_grade):
                # rejected_response = f'The correct answer is {option}. {options[option]} </s>'
                rejected_response = '''{"answer" : ''' + \
                    f'''"{option}. {options[option]}"''' + '''} </s>'''
                rejected_responses.append(rejected_response)

        student_code = codes[student_id]
        lora_prompts[student_id] = create_dpo_prompt(
            context, student_code, task, options)
        chosen[student_id] = chosen_response
        rejected[student_id] = rejected_responses

    # Split the dictionary into two lists based on the split parameter
    num_items = len(lora_prompts)
    split_idx = int(num_items * split)

    train_set = []
    test_set = []

    for idx, (key, value) in enumerate(lora_prompts.items()):
        prompt = format_user_prompt(lora_prompts[key], system_prompt)
        chosen_response = chosen[key]
        if (idx < split_idx):
            for rejected_response in rejected[key]:
                train_set.append(
                    {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response})
        else:
            for rejected_response in rejected[key]:
                test_set.append(
                    {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response})

    return train_set, test_set


def get_dpo_reasoning_dataset(context, codes, task, options, original_grades, original_reasonings, system_prompt, split=0.7):
    """
    Creates dataset for a single lab for finetuning with DPO

    Args : 
        context (str) : The simplified problem statement
        codes (dict) : A dictionary of student codes. Student ids are the keys
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
        original_grades (dict) : A dictionary of TA assigned grades. Student ids are the keys
        system_prompt (str) : The system prompt 
        split (float) : The train split for the dataset
    Returns : 
        A tuple of train and test splits. Both the splits are lists
        Each item in the split is a json with three fields "prompt", "chosen" and "rejected"
    """
    lora_prompts = {}
    chosen = {}
    rejected = {}

    for student_id in sorted(codes.keys()):
        if student_id not in original_grades:
            continue
        original_grade = original_grades[student_id]
        original_reasoning = original_reasonings[student_id]
        # chosen_response = f'The correct answer is {original_grade}. {options[original_grade]} </s>'

        chosen_response = '''{"answer" : ''' + \
            f'''"{original_grade}. {options[original_grade]} , "reasoning" : {original_reasoning}"''' + '''} </s>'''

        rejected_responses = []
        for option, rating in options.items():
            if (option != original_grade):
                # rejected_response = f'The correct answer is {option}. {options[option]} </s>'
                rejected_response = '''{"answer" : ''' + \
                    f'''"{option}. {options[option]} , "reasoning" : {rating}"''' + '''} </s>'''
                rejected_responses.append(rejected_response)

        student_code = codes[student_id]
        lora_prompts[student_id] = create_dpo_prompt(
            context, student_code, task, options)
        chosen[student_id] = chosen_response
        rejected[student_id] = rejected_responses

    # Split the dictionary into two lists based on the split parameter
    num_items = len(lora_prompts)
    split_idx = int(num_items * split)

    train_set = []
    test_set = []

    for idx, (key, value) in enumerate(lora_prompts.items()):
        prompt = format_user_prompt(lora_prompts[key], system_prompt)
        chosen_response = chosen[key]
        if (idx < split_idx):
            for rejected_response in rejected[key]:
                train_set.append(
                    {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response})
        else:
            for rejected_response in rejected[key]:
                test_set.append(
                    {"prompt": prompt, "chosen": chosen_response, "rejected": rejected_response})

    return train_set, test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--parent_dir", type=str,
                        help="The folder where all the labs are present")
    parser.add_argument("--eval_lab_names", nargs="+", default="",
                        help="List of the names of labs used for evaluation. All other labs will be used in the training split")
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Fraction of data to be used for training")
    parser.add_argument('--system_prompt_path', type=str, default="",
                        help="The file where system prompt is provided")
    parser.add_argument('--lora', type=int, default=1,
                        help="Create dataset for lora or dpo")
    parser.add_argument('--dpo_reasoning', type=int, default=0,
                        help="Create dpo dataset with reasoning")
    parser.add_argument('--shuffle', type=int, default=0,
                        help="Whether to shuffle the samples in the dataset")
    parser.add_argument('--verbose', type=int, default=0,
                        help="Whether to use the verbose problem statement")

    parser.add_argument('--train_dataset_path', type=str, default="",
                        help="Path to the train dataset")
    parser.add_argument('--test_dataset_path', type=str, default="",
                        help="Path to the evaluation dataset")

    args = parser.parse_args()

    # Extract system prompt
    with open(args.system_prompt_path, "r") as f:
        system_prompt = f.read().strip()

    train_dataset_file = open(args.train_dataset_path, "w")
    test_dataset_file = open(args.test_dataset_path, "w")

    train_points = []
    test_points = []

    for lab in os.listdir(args.parent_dir):
        current_lab_name = str(lab)
        lab_path = os.path.join(args.parent_dir, lab)

        train_split = args.train_split
        lab_grades_path = os.path.join(lab_path, "rubric_ratings.csv")

        if (os.path.isdir(lab_path)):
            if (lab in args.eval_lab_names):
                train_split = 0  # The entire lab will be used for evaluation if it's name is in args.eval_lab_names

            # Extract context (Simplified Problem Statement)
            if (args.verbose):
                ps_path = os.path.join(lab_path, "ps.txt")
                if os.path.exists(ps_path):
                    context_path = os.path.join(lab_path, "ps.txt")
                else:
                    context_path = os.path.join(
                        lab_path, "problemStatement.txt")
            else:
                context_path = os.path.join(lab_path, "modified_ps.txt")
            with open(context_path, "r") as f:
                context = f.read().strip()

            # Dict of student ids and their submissions
            student_submissions = get_submission_json(lab_path)

            # Get the criterion and rating descriptions
            all_criteria = get_rubrics(lab_path)

            # Get the original TA grades for that lab
            original_grades = extract_all_original_grades(lab_grades_path)
            original_reasonings = process_reasoning(lab_path)

            
            # Repeat for all criteria
            for criterion in all_criteria.keys():
                # Description for that particular criterion
                criterion_desc = all_criteria[criterion]["description"]
                # Rating descriptions for that particular criterion
                options = all_criteria[criterion]["ratings"]
                # Get the Grades And Rubrics related to specific criterion
                criterion_original_grades = original_grades[criterion]
                criterion_original_reasonings = original_reasonings[criterion]

                # For sentence output
                task = f'''Choose the option which is most suitable for the above code for the criterion "{criterion_desc}". Your response must start with "The correct answer is ". Strictly follow this output format at any cost'''
                
                # DPO Along with Reasoning
                if (args.dpo_reasoning):

                    train_set, test_set = get_dpo_reasoning_dataset(context, student_submissions, task, options,
                                                                    criterion_original_grades, criterion_original_reasonings, system_prompt, split=train_split)

                    for train_data_point in train_set:
                        train_points.append(train_data_point)

                    for test_data_point in test_set:
                        test_points.append(test_data_point)

                else:
                    train_set, test_set = get_dpo_dataset(context, student_submissions, task, options,
                                                            criterion_original_grades, system_prompt, split=train_split)

                    for train_data_point in train_set:
                        train_points.append(train_data_point)

                    for test_data_point in test_set:
                        test_points.append(test_data_point)

    # if (args.shuffle):
    #     random.shuffle(train_points)
    #     random.shuffle(test_points)
    
    for train_data_point in train_points:
        json.dump(train_data_point, train_dataset_file)
        train_dataset_file.write('\n')

    for test_data_point in test_points:
        json.dump(test_data_point, test_dataset_file)
        test_dataset_file.write('\n')

    train_dataset_file.close()
    test_dataset_file.close()
