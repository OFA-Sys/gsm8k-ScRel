
import json
import re
import os
from tqdm import tqdm, trange
from collections import Counter
import editdistance

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
TRAIN_FILE = '../data/train_use.jsonl'

def check_equation(string):
    if string.find('=') == -1:
        return False
    lhs = string.split('=')[0]
    rhs = string.split('=')[1]
    try:
        lhs_result = eval(str(lhs))
        if abs(float(lhs_result) - float(rhs)) < 1e-3:
            return True
    except BaseException:
        return False
    return False
    
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

def eval_json(json_path):
    with open(TRAIN_FILE, 'r') as f:
        lines = f.readlines()
    gold_ans = parse_gold(lines)

    with open(json_path, 'r') as f:
        lines = f.readlines()
    pred_ans = parse(lines)

    cor = 0
    for i in range(min(len(pred_ans), len(gold_ans))):
        if pred_ans[i] != INVALID_ANS and float(pred_ans[i]) == float(gold_ans[i]):
            cor += 1

    return {i:json.loads(lines[i])[0][1] for i in range(min(len(pred_ans), len(gold_ans)))}, \
           {i:pred_ans[i] for i in range(min(len(pred_ans), len(gold_ans)))}, \
           cor, min(len(pred_ans), len(gold_ans))

def barrier(counter):
    new_counter = dict()
    new_counter['0'] = counter[0]
    new_counter['1-20'] = sum({counter[i] for i in range(1,21)})
    new_counter['21-40'] = sum({counter[i] for i in range(21,41)})
    new_counter['41-60'] = sum({counter[i] for i in range(41,61)})
    new_counter['61-80'] = sum({counter[i] for i in range(61,81)})
    new_counter['81-99'] = sum({counter[i] for i in range(81,100)})
    new_counter['100'] = counter[100]
    return new_counter

def collect(folder, max_seed=100, temp='0.7'):
    print('---')
    print(folder)
    # path_list = [f'{folder}/raw_generation_train_{temp}_{idx}.json' for idx in range(0,max_seed+1,1)]
    path_list = [f'{folder}/raw_generation_{temp}_{idx}.json' for idx in range(0,max_seed,1)]
    path_list = [f for f in path_list if os.path.exists(f)]
    print(f'Generate seed count: {len(path_list)}')

    all_q = []
    all_ans = []
    new_path_list = []
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]
        res, pred, _, cnt = eval_json(file_path)
        if cnt > 7000:
            all_q.append(res)
            all_ans.append(pred)
            new_path_list.append(file_path)
    path_list = new_path_list

    with open(TRAIN_FILE, 'r') as f:
        lines = f.readlines()
    gold_ans = parse_gold(lines)

    sample_count_dict = {}
    problem_diffculty = []
    all_correct_cnt = 0
    pass_status = [False for _ in range(len(gold_ans))]
    pass_query = {idx:[] for idx in range(len(gold_ans))}
    for file_path_idx in range(len(path_list)):
        for idx in range(len(gold_ans)):
            if all_ans[file_path_idx][idx] != INVALID_ANS and float(all_ans[file_path_idx][idx]) == float(gold_ans[idx]):
                pass_status[idx] = True
                all_correct_cnt += 1
                problem_diffculty.append(idx)
                pass_query[idx].append(all_q[file_path_idx][idx])
        sample_count_dict[file_path_idx + 1] = sum(pass_status)
    print('Pass @ Sampling k seeds:')
    print(sample_count_dict)
    print('Pass count for all problems:')
    print(all_correct_cnt, len(path_list) * len(gold_ans))

    problem_diffculty_counter = Counter(problem_diffculty) # 1: 5, 2:10
    correct_pass_counter = Counter(problem_diffculty_counter.values())
    correct_pass_counter[0] = len(gold_ans) - sum(correct_pass_counter.values())
    correct_pass_counter_barrier = barrier(correct_pass_counter)
    print('Count of how often a problem is solved:')
    print(correct_pass_counter_barrier)
    
    equation_different_pass_query = {idx:[] for idx in range(len(gold_ans))}
    pattern = r'<<([^>]*)>>'  # Match everything inside << >>
    for idx in pass_query:
        if not pass_query[idx]:
            pass
        # matches = re.findall(pattern, json.loads(lines[idx])['response'])
        exist_match = []
        for q in pass_query[idx]:
            matches = re.findall(pattern, q)  # Find all matches
            equation_flag = True
            for match in matches:
                if not check_equation(match):
                    equation_flag = False
            if equation_flag:
                matches = '|'.join(matches).replace(' ', '')
                if matches not in exist_match:
                    equation_different_pass_query[idx].append(q)
                    exist_match.append(matches)
                else:
                    now_query_idx = exist_match.index(matches)
                    now_query = equation_different_pass_query[idx][now_query_idx]
                    now_score = sum([editdistance.eval(now_query, ref) for ref in equation_different_pass_query[idx] if ref != now_query])
                    q_score = sum([editdistance.eval(q, ref) for ref in equation_different_pass_query[idx] if ref != now_query])
                    if q_score > now_score:
                        equation_different_pass_query[idx][now_query_idx] = q

    equation_different_count = sum([len(equation_different_pass_query[idx]) for idx in equation_different_pass_query])        
    print(f'equation_different_count: {equation_different_count}')

    with open(f'{folder}/pass_query_{max_seed}_{temp}.json.v2.farest', 'w') as f:
        json.dump(pass_query, f, indent=4)

    with open(f'{folder}/pass_query_{max_seed}_{temp}_sft.jsonl.v2.farest', 'w') as f:
        for idx in pass_query:
            f.write(lines[idx].strip() + "\n")
            for q in pass_query[idx]:
                f.write(json.dumps({'query':json.loads(lines[idx])['query'], 'response':q}).strip() + "\n")

    with open(f'{folder}/equation_different_pass_query_{max_seed}_{temp}.json.v2.farest', 'w') as f:
        json.dump(equation_different_pass_query, f, indent=4)

    with open(f'{folder}/equation_different_pass_query_{max_seed}_{temp}_sft.jsonl.v2.farest', 'w') as f:
        for idx in equation_different_pass_query:
            f.write(lines[idx].strip() + "\n")
            for q in equation_different_pass_query[idx]:
                f.write(json.dumps({'query':json.loads(lines[idx])['query'], 'response':q}).strip() + "\n")


def collect_folders(folders, output_path, max_seed=100):
    print('---')
    path_list = []
    for folder in folders:
        print(folder)
        path_list.extend([f'{folder}/raw_generation_0.7_{idx}.json' for idx in range(0,max_seed+1,1)])

    path_list = [f for f in path_list if os.path.exists(f)]
    print(f'Generate seed count: {len(path_list)}')

    all_q = []
    all_ans = []
    for file_path_idx in range(len(path_list)):
        file_path = path_list[file_path_idx]
        res, pred, _, _ = eval_json(file_path)
        all_q.append(res)
        all_ans.append(pred)

    with open(TRAIN_FILE, 'r') as f:
        lines = f.readlines()
    gold_ans = parse_gold(lines)

    sample_count_dict = {}
    problem_diffculty = []
    all_correct_cnt = 0
    pass_status = [False for _ in range(len(gold_ans))]
    pass_query = {idx:[] for idx in range(len(gold_ans))}
    for file_path_idx in range(len(path_list)):
        for idx in range(len(gold_ans)):
            if all_ans[file_path_idx][idx] != INVALID_ANS and float(all_ans[file_path_idx][idx]) == float(gold_ans[idx]):
                pass_status[idx] = True
                all_correct_cnt += 1
                problem_diffculty.append(idx)
                pass_query[idx].append(all_q[file_path_idx][idx])
        sample_count_dict[file_path_idx + 1] = sum(pass_status)
    print('Pass @ Sampling k seeds:')
    print(sample_count_dict)
    print('Pass count for all problems:')
    print(all_correct_cnt, len(path_list) * len(gold_ans))

    problem_diffculty_counter = Counter(problem_diffculty) # 1: 5, 2:10
    correct_pass_counter = Counter(problem_diffculty_counter.values())
    correct_pass_counter[0] = len(gold_ans) - sum(correct_pass_counter.values())
    correct_pass_counter_barrier = barrier(correct_pass_counter)
    print('Count of how often a problem is solved:')
    print(correct_pass_counter_barrier)
    
    equation_different_pass_query = {idx:[] for idx in range(len(gold_ans))}
    pattern = r'<<([^>]*)>>'  # Match everything inside << >>
    for idx in pass_query:
        if not pass_query[idx]:
            pass
        # matches = re.findall(pattern, json.loads(lines[idx])['response'])
        exist_match = []
        for q in pass_query[idx]:
            matches = re.findall(pattern, q)  # Find all matches
            equation_flag = True
            for match in matches:
                if not check_equation(match):
                    equation_flag = False
            if equation_flag:
                matches = '|'.join(matches).replace(' ', '')
                if matches not in exist_match:
                    equation_different_pass_query[idx].append(q)
                    exist_match.append(matches)
                else:
                    now_query_idx = exist_match.index(matches)
                    now_query = equation_different_pass_query[idx][now_query_idx]
                    now_score = sum([editdistance.eval(now_query, ref) for ref in equation_different_pass_query[idx] if ref != now_query])
                    q_score = sum([editdistance.eval(q, ref) for ref in equation_different_pass_query[idx] if ref != now_query])
                    if q_score > now_score:
                        equation_different_pass_query[idx][now_query_idx] = q

    equation_different_count = sum([len(equation_different_pass_query[idx]) for idx in equation_different_pass_query])        
    print(f'equation_different_count: {equation_different_count}')

    with open(f'{folder}/pass_query_{max_seed}.json.v2.farest', 'w') as f:
        json.dump(pass_query, f, indent=4)

    with open(f'{folder}/pass_query_{max_seed}_sft.jsonl.v2.farest', 'w') as f:
        for idx in pass_query:
            f.write(lines[idx].strip() + "\n")
            for q in pass_query[idx]:
                f.write(json.dumps({'query':json.loads(lines[idx])['query'], 'response':q}).strip() + "\n")

    with open(f'{folder}/equation_different_pass_query_{max_seed}.json.v2.farest', 'w') as f:
        json.dump(equation_different_pass_query, f, indent=4)

    with open(output_path, 'w') as f:
        for idx in equation_different_pass_query:
            f.write(lines[idx].strip() + "\n")
            for q in equation_different_pass_query[idx]:
                f.write(json.dumps({'query':json.loads(lines[idx])['query'], 'response':q}).strip() + "\n")

if __name__ == "__main__":
    collect('./ckpts/gsm8k_sft_llama7b', 100)
    