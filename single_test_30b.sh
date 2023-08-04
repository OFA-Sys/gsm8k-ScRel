MODEL_DIR=$1
# mkdir -p $OUT_DIR
CUDA_VISIBLE_DEVICES=$2 python single_inference_30b.py \
        --base_model $MODEL_DIR \
        --data_path $3 \
        --out_path $MODEL_DIR \
        --batch_size 1 \
        --test_shard $2
