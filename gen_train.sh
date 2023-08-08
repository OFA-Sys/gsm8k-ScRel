MODEL_DIR=$1
DATA_DIR=$2
SEED=$3
torchrun --nproc_per_node 8 --master_port 7834 single_inference_7b_13b.py \
                        --base_model $MODEL_DIR \
                        --data_path $DATA_DIR \
                        --out_path $MODEL_DIR \
                        --batch_size 8 \
                        --seed $SEED
