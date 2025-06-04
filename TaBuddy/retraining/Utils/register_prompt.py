import mlflow
from datetime import datetime

print(mlflow.__version__)
current_date = datetime.today().strftime('%Y-%m-%d')

# Double curly braces are required for prompt variable placeholders
grading_prompt_template = """<s>[INST] <<SYS>>
Your task is to choose the MOST suitable option among a set of options I provide, about a code which will also be provided. Give your output as a json with a single field "answer". Do not output anything else. Strictly follow this output format at any cost.
<</SYS>>

### Context : 
{{ context }}

### Code : 
{{ code }}

### Task :
{{ task }}

### Options :
{{ options }}

### Response : The required output in json format is : [/INST]"""

# Register the prompt template
prompt = mlflow.register_prompt(
    name="DPO_Prompt",
    template=grading_prompt_template,
    commit_message="Prompt For Fine-Tuning Using DPO",
    version_metadata={
        "author": "saurav@cse.iitb.ac.in",
        "date": current_date,
    },
    tags={
        "task": "grading",
        "model_type": "DPO",
        "language": "cpp",
    },
)

print(f"✅ Created prompt '{prompt.name}' (version {prompt.version})")
