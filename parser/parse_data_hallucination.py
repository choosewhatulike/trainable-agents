import json
import os
import sys
from collections import defaultdict
import re

data_path = sys.argv[1]
character = data_path.split('-char-')[1].replace('.jsonl', '')
data_time = re.findall(r'\d+-\d+-\d+', data_path)[0]
out_path = f'./processed/{data_time}/generated_agent_hallucination_{character}.json'

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

def parse_dialogue_info(text):
    res = []
    parts = list(text.split('\n\n'))
    prefix = []
    for idx, p in enumerate(parts):
        if 'Background:' in p:
            prefix.append(p)
            continue
        role = ''
        action = ''
        is_dialogue = False
        lines = list(p.split('\n'))
        if len(lines) == 3:
            role = lines[0].strip()
            action = lines[1].strip()
            content = lines[2].strip()
            res.append({
                'role': role,
                'action': action,
                'content': content.strip('"')
            })
        elif len(lines) == 2:
            t = lines[0].strip()
            if '(' in t or ')' in t:
                is_dialogue = True
            elif ':' in t:
                is_dialogue = True
            elif t.upper() == t:
                is_dialogue = True
            if is_dialogue:
                lpos = t.find('(')
                rpos = t.find(')')
                if lpos >= 0 and rpos >= 0:
                    role = t[:lpos] + t[rpos+1:]
                    action = t[lpos:rpos+1]
                else:
                    role = t.split(':')[0].strip()
                content = lines[1].strip()
                res.append({
                        'role': role,
                        'action': action,
                        'content': content.strip('"')
                    })
        elif len(lines) == 1:
            t = lines[0].strip()
            if '(' in t or ')' in t:
                is_dialogue = True
            elif ':' in t:
                is_dialogue = True
            if is_dialogue:
                lpos = t.find('(')
                rpos = t.find(')')
                if lpos >= 0 and rpos >= 0:
                    role = t[:lpos]
                    action = t[lpos:rpos+1]
                    content = t[rpos+1:].strip().strip(':')
                else:
                    role = t.split(':')[0].strip()
                    content = t.split(':')[1].strip()
                res.append({
                        'role': role,
                        'action': action,
                        'content': content.strip('"')
                    })
        
    new_res = []
    for ex in res:
        if not ex['role'] or not ex['content']:
            continue
        ex['role'] = re.sub(r'\((.*?)\)', '', ex['role'])
        ex['role'] = ex['role'].replace(':', '').strip()
        ex['content'] = re.sub(r'\((.*?)\)', '', ex['content']).strip()
        # ex['content'] = ex['content'].replace(':', '').strip()
        new_res.append(ex)
    res = new_res
    new_prefix = []
    for p in prefix:
        p = p.replace('Background:', '').strip()
        new_prefix.append(p)
    prefix = new_prefix
    if not res:
        return 'EPT'
    if len(res) < 3:
        return 'SHT'
    if not prefix:
        return 'PRE'
    return {'setting': prefix, 'dialogue': res}


counter = defaultdict(int)
raw_data = load_gen_data(data_path)
results = []
for ex in raw_data:
    id = ex['gen_answer_id']
    prompt = ex['prompt']
    lpos = prompt.find('- Location: ')
    loc = prompt[lpos:].split('\n')[0].replace('- Location: ', '')
    bpos = prompt.find('- Background: ')
    back = prompt[bpos:].split('\n')[0].replace('- Background: ', '')
    out = parse_dialogue_info(ex['completions'])
    if isinstance(out, str):
        counter[out] += 1
        print(id)
        continue
    out['location'] = loc
    out['background'] = back
    out['source'] = f'seed_dialogue_{id}'
    results.append(out)

print("output_path", out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2)

print('final length', len(results))
print(counter)

sum_turns = 0
sum_turn_words = 0
for r in results:
    sum_turns += len(r['dialogue'])
    for d in r['dialogue']:
        sum_turn_words += len(d['content'].split())
print(f'total turn:{sum_turns}, avg turn:{sum_turns/len(results)}, total words: {sum_turn_words}, avg words: {sum_turn_words/sum_turns}')
