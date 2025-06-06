# Large Language Model Fine Tuning

Large Language Model is a complex statistical model, primarily based on neural networks and the 'transformer' architecture, that learns patterns within large datasets. 

Key points : 

1. Vector Representation : Words are represented as vectors in a high-dimensional space, where similar words are positioned closed together, enabling the model to understand semantic. 
2. Attention Mechanism : A core component of transformers, where the model focuses on specific parts of the input sequence to better understand the context and generate relevant output. 
3. Loss Function : A mathematical function used to measure the difference between the model's predicted output and the actual target, guiding the learning process through backpropagation. 
4. Gradient Descent : An optimization algorithm that iteratively adjusts the model parameters to minimize the loss function, leading to improved predictions.

Fine-tuning large language models (LLMs) is a powerful technique that allows us to adapt pre-trained models to specific tasks or domains. This enables us to leverage the knowledge embedded within these models and tailor them to a wide range of applications. In this context, we will explore how to fine-tune pre-trained models, such as CodeLLama and CodeStral, to automate the grading of programming assignments.

![Fine Tuning](img/finetuning.jpeg)

Illustration of the fine-tuning process for adapting pre-trained models like CodeLLama and CodeStral to specific tasks, such as automating the grading of programming assignments.



The figure below gives a brief information about the task for which we perform the finetuning of large language model.

![Task](img/Task.jpg)

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone "https://github.com/sauravchaudharysc/LLM-Finetuning.git"
   ```

2. Navigate to the project directory:

   ```bash
   cd LLM-Finetuning
   ```

3. Create a virtual environment (optional but recommended):

   - **For Windows**:

     ```bash
     python -m venv venv
     ```

   - **For macOS/Linux**:

     ```bash
     python3 -m venv venv
     ```

4. Activate the virtual environment:

   - **For Windows**:

     ```bash
     .\venv\Scripts\activate
     ```

   - **For macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

5. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. You’re ready to go! You can now run your project.

   

## Dataset Creation for Fine-Tuning

This repository contains a Python script that automates the creation of datasets for fine-tuning a model using a [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) approach. The datasets consist of student submissions to various problem statements, and the grades assigned by Teaching Assistants (TAs). The dataset generation script processes the students' submissions and the associated grading rubrics to create a train and test dataset for training.

The dataset creation process is executed through the `create_dataset.py` script located in `Utils/`, which can be invoked using a bash script `create_dpo_dataset.sh` . 

### Directory Structure

Before running the script, ensure the following directory structure:

```
parent_dir/
    └── lab_name/
        ├── ps.txt (or modified_ps.txt or problemStatement.txt)
        ├── rubric_ratings.csv
        ├── rubrics.json
        ├── submissions/
        └── other lab-related files
```

### Input Parameters

To generate the datasets, provide the following parameters:

- `--parent_dir`: Path to the directory containing all the labs.
- `--eval_lab_names`: List of lab names to be used for evaluation (i.e., the dataset completely shielded).
- `--train_split`: Fraction of data used for training. Default is 0.7 (70% training, 30% testing).
- `--system_prompt_path`: Path to the file containing the system prompt.
- `--train_dataset_path`: Path to the file where the training dataset should be saved (JSONL format).
- `--test_dataset_path`: Path to the file where the testing dataset should be saved (JSONL format).
- `--verbose`: A flag to use a verbose problem statement (0 for non-verbose, 1 for verbose).

```bash
bash create_dpo_dataset.sh
```

This script will automatically call `create_dataset.py` and create the dataset in the specified paths.

## Fine-Tuning 

This script fine-tunes a pre-trained language model using [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) and [Low-Rank Adaptation (LoRA)](https://arxiv.org/pdf/2106.09685). It allows users to specify the model size, device, LoRA configuration, and dataset paths through command-line arguments. 

Key Steps:

1. **Model Initialization**: Loads the model and tokenizer, optionally with quantization and LoRA adapters.
2. **Training Setup**: Configures training parameters like batch size, learning rate, and number of epochs.
3. **Training**: The model is trained using the `DPOTrainer`, with training and evaluation datasets loaded from JSONL files.
4. **Saving**: After training, the model is saved with a timestamped directory, and the training time and data loading time are recorded in a `time.txt` file.

The `dpo_train.sh` script will execute the Python fine-tuning command with the specified model size, dataset paths, and output directory. Ensure the dataset paths are correct.

```bash
bash dpo_train.sh	
```

