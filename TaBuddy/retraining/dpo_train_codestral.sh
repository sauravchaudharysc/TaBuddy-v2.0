CUDA_VISIBLE_DEVICES="0" python3 Utils/dpo_train_codestral.py \
    --model_size "22b" \
    --output_dir "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Model/CodeStral" \
    --train_dataset_path "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Dataset/train.jsonl" \
    --eval_dataset_path "/raid/ganesh/nagakalyani/nagakalyani/siamese/Saurav_Experiments/Dataset/test.jsonl"  
