export model_path=$1
nohup bash ./single_test_30b.sh $model_path 0 "./data/test_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 1 "./data/test_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 2 "./data/test_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 3 "./data/test_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 4 "./data/test_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 5 "./data/test_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 6 "./data/test_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 7 "./data/test_use.jsonl" >/dev/null 2>&1 &



while true
do
    sum=0
    for i in {0..7}
    do
        if [ -e $model_path/raw_generation_greedy_shard_${i}.json ];then
        ((sum+=1))
        else
        echo $model_path/raw_generation_greedy_shard_${i}.json
        echo "not finish"
        fi
    done

    if [ $sum = 8 ];then
    break
    else
    sleep 30s
    fi
done
