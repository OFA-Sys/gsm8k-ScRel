export MASTER_ADDR=localhost
export MASTER_PORT=2131
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_DIR=$1
OUT_DIR=$1

torchrun --nproc_per_node 8 --master_port 7834 test.py \
                        --base_model $MODEL_DIR \
                        --data_path "./data/test_use.jsonl" \
                        --out_path $OUT_DIR \
                        --batch_size 6