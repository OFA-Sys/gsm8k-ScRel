MODEL_PATH=your_llama-30b
SAVE_PATH=$2

export master_addr=${MASTER_ADDR}
export master_port=${MASTER_PORT}
export local_rank=${RANK}

wandb disabled
torchrun --nproc_per_node=8 --master_port=$master_port --nnode=2 --node_rank=$RANK --master_addr=$MASTER_ADDR train_llama_30b_65b.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $1 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs $3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed config/zero2_config_30b.json