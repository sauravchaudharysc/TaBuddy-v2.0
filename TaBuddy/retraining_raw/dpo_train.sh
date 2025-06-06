CUDA_VISIBLE_DEVICES="6" python3 Utils/dpo_train_tushar.py \
    --model "CodeLLama-7b" \
    --output_dir "./Model/CodeLlama" \
    --train_dataset_path "./Dataset/train.jsonl" \
    --eval_dataset_path "./Dataset/eval.jsonl" \
    --test_dataset_path "./Dataset/test.jsonl" 