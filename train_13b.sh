export MODEL_PATH='your_llama_13b/13b2'
export SAVE_PATH=$2
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true
wandb offline

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $1 \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs $3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 174 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True