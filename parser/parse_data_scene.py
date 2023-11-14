import json
import os
import sys
from collections import defaultdict
import re

data_path = sys.argv[1]
character = data_path.split('-char-')[1].replace('.jsonl', '')
data_time = re.findall(r'\d+-\d+-\d+', data_path)[0]
out_path = f'./processed/{data_time}/generated_agent_scene_{character}.json'
need_print = True

def load_gen_data(path):
    with open(path, 'r', encoding='utf-8') as fp:
        raw = fp.read().split('}{\n')
    data = []
    for s in raw:
        s = s.strip()
        if not s.startswith('{'):
            s = '{' + s
        if not s.endswith('}'):
            s = s + '}'
        ex = json.loads(s)
        data.append(ex)
    return data


def parse_scene_info(text):
    res = {}
    lines = text.strip().split('\n')
    for line in lines:
        if 'Type: ' in line:
            res['type'] = line.split('Type: ')[1].strip()
        if 'Location: ' in line:
            res['location'] = line.split('Location: ')[1].strip()
        if 'Background: ' in line:
            res['background'] = line.split('Background: ')[1].strip()
    if 'type' not in res:
        if len(lines) == 3:
            type_str = lines[0].split(': ')
            if len(type_str) == 2:
                type_str = type_str[1].strip()
            else:
                type_str = ''
            if type_str:
                res['type'] = type_str
    if len(res) != 3:
        return 'INV'
    return res


counter = defaultdict(int)
raw_data = load_gen_data(data_path)
results = []
for ex in raw_data:
    if not ex['check_result']:
        continue
    id = ex['gen_answer_id']
    cid = 1
    profile = ex['prompt'].split('Imagine 20 scenes that')[0].replace('Context:', '').strip()
    for t in ex['completions'].split('\n\n'):
        out = parse_scene_info(t)
        if isinstance(out, str):
            if need_print:
                print(ex['ID'])
            counter[out] += 1
            continue
        out['source'] = f'seed_scene_{id}_c{cid}'
        out['profile'] = profile
        cid += 1
        results.append(out)

print("output_path=", out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2)

print(len(results))
print(counter)
