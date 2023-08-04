import json
import re
import os
from collections import Counter

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    if completion.find('\u0000') >= 0:
        completion = completion[0:completion.find('\u0000')]
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            float(match_str)
        except BaseException:
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS

def parse_gold(lines):
    all_ans = []
    for line in lines:
        try:
            ans = extract_answer(json.loads(line)['response'])
        except BaseException:
            print(line)
            ans = extract_answer(json.loads(line)['answer'])
        all_ans.append(ans)
    return all_ans

def parse(lines):
    all_ans = []
    for line in lines:
        try:
            ans = extract_answer(json.loads(line)[0][1])
        except BaseException:
            ans = extract_answer(json.loads(line)['gen'][0])
        all_ans.append(ans)
    return all_ans

def eval_json(json_path, mode='test'):
    if json_path.endswith('/') or not json_path.endswith('json'):
        origin_json_path = json_path
        json_path = os.path.join(json_path, 'raw_generation_greedy.json')
    if not os.path.exists(json_path):
        lines = []
        for i in range(8):
            path = os.path.join(origin_json_path, f'raw_generation_greedy_shard_{i}.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    now_lines = f.readlines()
                lines.extend(now_lines)

        if not lines:
            for i in range(8):
                path = os.path.join(origin_json_path, f'raw_generation_greedy_on_{mode}_shard_{i}.json')
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        now_lines = f.readlines()
                    lines.extend(now_lines)
    else:
        with open(json_path, 'r') as f:
            lines = f.readlines()
   
    pred_ans = parse(lines)

    if not pred_ans:
        return

    with open(f'./data/{mode}_use.jsonl', 'r') as f:
        lines = f.readlines()
    gold_ans = parse_gold(lines)

    cor = 0
    rg = range(min(len(pred_ans), len(gold_ans)))
    for i in rg:
        if pred_ans[i] != INVALID_ANS and abs(float(pred_ans[i]) - float(gold_ans[i])) < 1e-4:
            cor += 1
    print(json_path, cor, cor/len(list(rg)) * 100, len(rg))
    return pred_ans


def eval_majority_voting(folder, max_cnt=100):
    lines = []
    paths = [os.path.join(folder, f'raw_generation_0.7_{i}_test.json') for i in range(max_cnt)]
    idx = 0
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                now_lines = f.readlines()
            lines.append(now_lines)
            idx += 1
    if not lines:
        paths = [os.path.join(folder, f'raw_generation_0.7sampled_on_test_seed_{i}_shard_SHARD.json') for i in range(max_cnt)]
        for path in paths:
            now_lines = []
            for SHARD in range(8):
                p = path.replace('SHARD', str(SHARD))
                if os.path.exists(p):
                    with open(p, 'r') as f:
                        now_shard_lines = f.readlines()
                        now_lines.extend(now_shard_lines)
            if len(now_lines) == 1319:
                lines.append(now_lines)
                idx += 1
    if not lines:
        return

    def maj(lst):
        lst = [x for x in lst if x != INVALID_ANS]
        if not lst:
            return INVALID_ANS
        # Count the occurrences of each string in the list
        counts = Counter(lst)
        
        # Find the string with the highest count
        most_common = max(counts, key=counts.get)
        
        return most_common

    pred_ans_multiple = [parse(prediction) for prediction in lines]
    pred_ans = [maj([prediction[i] for prediction in pred_ans_multiple]) for i in range(len(pred_ans_multiple[0]))]

    if not pred_ans:
        return

    with open(f'./data/test_use.jsonl', 'r') as f:
        lines = f.readlines()
    gold_ans = parse_gold(lines)


    cor = 0
    rg = range(min(len(pred_ans), len(gold_ans)))
    for i in rg:
        if pred_ans[i] != INVALID_ANS and abs(float(pred_ans[i]) - float(gold_ans[i])) < 1e-4:
            cor += 1
    print(folder, cor, cor/len(list(rg)) * 100, len(rg), f'Ensemble count: {idx}')
    return pred_ans

if __name__ == "__main__":

    eval_json('./ckpts/gsm8k_sft_llama7b/raw_generation_greedy.json', 'test')
