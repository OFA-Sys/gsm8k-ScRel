for seed in $(seq 1 100)
do
    echo $seed
    bash ./gen_train.sh $1 ./data/train_use.jsonl $seed
done
