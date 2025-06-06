import multiprocessing as mp

# Set 'spawn' method for Celery worker processes to avoid CUDA re-initialization errors
mp.set_start_method('spawn', force=True)

import random
import time
import torch
import ast
import os
from django.conf import settings



def json_from_string (string) : 
    return ast.literal_eval(string.strip())

def truncate_to_100_words(text):
    words = text.split()
    if len(words) > 100:
        return ' '.join(words[:100])
    return text


def extract_llm_ratings (lab_results_path, criterion_name, criterion_responses="") :
    predicted_results = {}

    # LLM outputss
    
    count = 0
    print(f'The criterion response is {criterion_responses}')
    for student_id, model_response in criterion_responses.items():
        stripped_model_response = model_response.strip()
        start_index = model_response.find('{')
        end_index = model_response.find('}') + 1

        content_within_braces = model_response[start_index:end_index]
        already_extracted = 1
        try : 
            
            extracted_ans = json_from_string(content_within_braces)
            already_extracted = 0
        except :
            if (stripped_model_response.startswith('''{\n"answer": "''')) :
                option = stripped_model_response[13]
            elif (stripped_model_response.startswith('''{\"answer\" : ''')) :
                option = stripped_model_response[12] 
            elif (stripped_model_response.startswith("The correct answer is ")) : 
                option = stripped_model_response[22]
            elif (stripped_model_response.startswith("Answer: ")):
                option = stripped_model_response[8]
            else : 
                count += 1
                    # print(student_id, model_response)
                continue
        reasoning="I am unable to provide the reasoning for this criterion."        
        if not (already_extracted) : 
            try:
                option = extracted_ans['answer'][0]
            except Exception as e:
                continue
            try:
                reasoning = extracted_ans['reasoning']
            except Exception as e:
                continue
        try : 
            option = option.capitalize()
        except Exception as e : 
            pass
            
        diff = ord(option) - ord('A')
        if not(diff >= 0 and diff < 4) : 
            # print(student_id, model_response[:20])
            continue
        reasoning = truncate_to_100_words(reasoning)    
        result=[]
        result.append(option)
        result.append(reasoning)
        predicted_results[student_id] = result
       

    return predicted_results



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

def generate_single_response(model, tokenizer, user_prompt, device, max_length=1024, system_prompt=""):

    """
    Generates a response for a single user prompt.

    Args : 
        model : The model which has been loaded into memory
        tokenizer : The tokenizer which has been loaded into memory
        user_prompt (str) : The user prompt
        max_length (int) : The maximum input length
        system_prompt (str) : The system prompt
        device (str) : The device on which the inference is going to run 

    Returns : 
        A string response from the model
    """
    formatted_prompt = format_user_prompt(user_prompt, system_prompt=system_prompt)


    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=max_length ,add_special_tokens=False).to(device)

    output = model.generate(
        **inputs,
        # attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.1,
        temperature=0.1,
        max_new_tokens=512
    )

    # Extract the new tokens (response) from the generated tokens.
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response

def create_zero_shot_prompt (context, code, task, options) : 
    """
    Creates a zero shot user prompt.

    Args : 
        context (str) : The simplified problem statement
        code (str) : The student code
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")
    Returns : 
        A zero shot user prompt as a string
    """
    options_list = ""
    for key in sorted(options.keys()) : 
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

    # prompt += '''{"answer" : '''
    return prompt


def create_zero_shot_prompts (context, codes, task, options) :
    """
    Create zero-shot prompts for a set of student submissions.
        
    Args : 
        context (str) : Modified problem statement
        codes (dict) : A dictionary of all the student codes. Keys are the student ids
        task (str) : The task description, i.e, what the model has to do (Can be similar to system prompt)
        options (dict) : A dictionary of option names(eg. "A", "B", "C", ..) and their descriptions (eg. "Good variable names", "Poor variable names")

    Returns : 
        A dictionary of zero-shot user prompts for all the students. Keys are the student ids
    """
    
    zero_shot_prompts = {}
    file_path =  os.path.join(settings.LOG_DIR,'Inference', 'prompt.txt')
    for student_id in sorted(codes.keys()) : 
        student_code = codes[student_id]

        zero_shot_prompts[student_id] = create_zero_shot_prompt(context, student_code, task, options)
       
        with open(file_path, "a",  encoding="utf-8") as f:
            f.write(zero_shot_prompts[student_id] + "\n\n\n")  


    return zero_shot_prompts


def grade_k_shot (model, tokenizer, system_prompt, zero_shot_prompts, device, max_length=1024, text_dump=False) :

    '''
    Grades student submissions using zero-shot and few-shot prompting.

    Args : 
        model : The model which has been loaded into memory
        tokenizer : The tokenizer which has been loaded into memory
        system_prompt (str) : The system prompt which will be used for grading all submissions
        zero_shot_prompts (dict) : A dictionary of user prompts (0-shot or few-shot). Student ids are the keys
        output_file_path (str) : The path to the file to print all the model responses
        device (str) : The device where the grading will be done
        max_length (int) : The maximum input length. Rest of the input will be truncated

    Returns : 
        None 
    ''' 
    responses = {}

    for student_id in sorted(zero_shot_prompts.keys()) :
        user_prompt = zero_shot_prompts[student_id]
        
        string_response = generate_single_response(model, tokenizer, user_prompt, device, system_prompt=system_prompt, max_length=max_length)
        responses[student_id] = string_response
        
    return responses

def grade_submissions(
    tokenizer, model, device, problem_statement, submissions, criterion_info, criterion_name="", max_length=4096, few_shot=False, few_shot_examples=0, train_split=0.7
):
    torch.manual_seed(0)
    random.seed(0)

    start_time = time.time()

    # Extract student submissions
    student_submissions = submissions
    student_submissions_copy = {}
    sorted_student_ids = sorted(student_submissions.keys())
    for i in range(len(student_submissions)):
        student_id = sorted_student_ids[i]
        student_submissions_copy[student_id] = student_submissions[student_id]
    student_submissions = student_submissions_copy

    # Extract system prompt
    system_prompt = '''Your task is to choose the MOST suitable option among a set of options I provide, about a code which will also be provided. Give your output as a json with a single field "answer". Do not output anything else. Strictly follow this output format at any cost.'''

    # Extract the context (Simplified problem statement)
    context = problem_statement

    # Extract criterions
    criterion_descs = []
    criterion_ids = []
    options_list = []

    for criterion_obj in criterion_info:
        options = {}
        criterion_id = criterion_obj["id"]
        criterion_desc = criterion_obj["description"]
        raw_options = criterion_obj["Ratings"]

        for json_obj in raw_options:
            options[json_obj["title"]] = json_obj["description"]

        criterion_ids.append(criterion_id)
        criterion_descs.append(criterion_desc)
        options_list.append(options)


    outputs = {}
    for idx in range(len(criterion_descs)):
        criterion_desc = criterion_descs[idx]
        criterion_id = str(criterion_ids[idx])
        options = options_list[idx]
        if not few_shot:
            # This describes the task for the LLM
            task = f'''Choose the option which is most suitable for the above code for the criterion "{criterion_desc}". Give your output as a json with two fields : "answer" and "reasoning". Do not output anything else. Strictly follow this output format.'''
            zero_shot_prompts = create_zero_shot_prompts(context, student_submissions, task, options)
            response = grade_k_shot(model, tokenizer, system_prompt, zero_shot_prompts, device, max_length=max_length)
            outputs[criterion_id] = response
    end_time = time.time()
    print("Total time taken :", end_time - start_time)
    return outputs
