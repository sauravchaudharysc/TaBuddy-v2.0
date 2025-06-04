python3 Utils/create_dataset.py \
    --parent_dir /raid/ganesh/nagakalyani/nagakalyani/siamese/dataset27_modified \
    --eval_lab_names "CP_00214_var_cs101s23_lq01_d_q2" "CP_00206_var_cs101s23_lq01_b_q2" "cs101a23_lq01_a_q2" "cs101a23_lq01_d_q3" "CP_00116_sort_cs101f22_LE03_E_Q1" "CP_00115_sort_cs101f22_LE03_C_Q1" \
    --system_prompt_path ./Utils//dpo_sys_prompt.txt \
    --train_dataset_path ./Dataset/train.jsonl \
    --test_dataset_path ./Dataset/test.jsonl \
    --verbose 0