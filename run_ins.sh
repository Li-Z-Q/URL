base_model=Baichuan2-7B-Chat
num_train_epochs=2
learning_rate=1e-4 

train_data=manual_1K_ins
# train_data=manual_1K_no_ins
# train_data=manual_1K_single_ins

gradient_accumulation_steps=16 
save_steps=40

loss_rate=0.2
# loss_rate=0.0

gist_token=#-以-上-内-容-的-总-结-#
prompt_id=prompt0
prompt="instance-specific"

export WANDB_API_KEY=XXX
export WANDB_PROJECT=llm_matching_ins
export WANDB_RUN_NAME=${base_model}_${train_data}_${loss_rate}_${num_train_epochs}_${learning_rate}_${gradient_accumulation_steps}_${prompt_id}_${gist_token}
export PYTHONPATH="/shared_home/XXX/llm-matching"

echo "WANDB_RUN_NAME: ${WANDB_RUN_NAME}"

deepspeed --master_port=1225 run_ins.py \
    --output_dir /ceph_home/XXX/llm_matching/${WANDB_PROJECT}/${WANDB_RUN_NAME} \
    --model_name_or_path /ceph_home/XXX/${base_model} \
    --train_data /ceph_home/XXX/llm_matching/${train_data}.jsonl \
    --learning_rate ${learning_rate} \
    --fp16 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --logging_steps 1 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.8 \
    --query_max_len 256 \
    --passage_max_len 256 \
    --train_group_size 2 \
    --negatives_cross_device \
    --gist_token ${gist_token} \
    --loss_rate ${loss_rate} \
    --prompt ${prompt} \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 20 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target W_pack \
    --q_lora False \
    --overwrite_output_dir \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --warmup_ratio 0.08 \