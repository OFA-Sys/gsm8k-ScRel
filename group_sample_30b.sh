export model_path=$1
export temp=$2
export seed_range=100
nohup bash ./single_test_30b.sh $model_path 0 $temp $seed_range "./data/train_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 1 $temp $seed_range "./data/train_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 2 $temp $seed_range "./data/train_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 3 $temp $seed_range "./data/train_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 4 $temp $seed_range "./data/train_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 5 $temp $seed_range "./data/train_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 6 $temp $seed_range "./data/train_use.jsonl" >/dev/null 2>&1 &
nohup bash ./single_test_30b.sh $model_path 7 $temp $seed_range "./data/train_use.jsonl" >/dev/null 2>&1 &


while true
do
    sum=0
    for i in {0..6}
    do
        ((sum+=1))
    done

    if [ $sum = 8 ];then
    break
    else
    sleep 30s
    fi
done