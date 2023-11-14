from eval_utils import decoder_for_openai, get_character_names, seed_data_dir
from io_utils import read_profile, read_json, read_gen_data
import os
import glob
from threading import Thread, Lock
import json
from functools import partial
import re
from tqdm import tqdm
import sys

gen_results_dir = './evaluation_result/interview_single'
output_dir = './evaluation_result/score_chatgpt_single'
aspect = sys.argv[1]
assert aspect in ['memory', 'values', 'personality', 'hallucination'], f'aspect:{aspect} is not supported!'
DEBUG = True

if DEBUG:
    names = ['Beethoven']
else:
    names = get_character_names()

prompt_ds = []
n_workers = 20
current_idx = 0

with open(os.path.join(seed_data_dir, 'prompts', f'prompt_score_llm_{aspect}.txt'), 'r', encoding='utf-8') as fp:
    meta_prompt = fp.read().strip()
    

def get_reply(d_list):
    result = ''
    if isinstance(d_list, dict):
        result = d_list['content']
    else:
        for d in d_list:
            if d['action'] == '(speaking)':
                result = d['content']
                break
    result = result.split('\n\n')[0]
    result = re.sub(r'\*.*?\*', '', result)
    return result
    
def get_prompt_item(name, profile, ds, meta_prompt, model_prefix, model_result_path):
    prompt_ds = []
    loc_time = 'Coffee Shop - Afternoon'
    status = f'{name} is casually chatting with a man from the 21st century. {name} fully trusts the man who engage in conversation and shares everything {name} knows without reservation.'
    role = 'Man'
    
    for idx, ex in enumerate(ds):
        question = ex['question']
        reply = get_reply(ex['reply'])
        context_str = profile[0]
        interaction_str = f'{role}: {question}\n{name}: {reply}'
        prompt = meta_prompt.format(
                agent_name=name,
                agent_context=context_str,
                loc_time=loc_time,
                status=status,
                interactions=interaction_str,
            )
        prompt_ds.append({
            'prompt': prompt,
            'model_name': model_prefix,
            'answer_path': f'{model_result_path}-id-{idx}',
            'question': question,
            'qid': idx,
            })
    return prompt_ds
    
prompt_ds = []
for name in names:
    _, profile = read_profile(os.path.join(seed_data_dir, 'profiles',  f'wiki_{name}.txt'))
    for model_result_dir in os.listdir(gen_results_dir):
        if not model_result_dir.startswith(name):
            continue
        model_name = model_result_dir.replace(f'{name}_', '').replace('_result', '')
        result_path = sorted(glob.glob(os.path.join(gen_results_dir, model_result_dir, '*.json')))[0]
        ds = read_json(result_path)
        prompt_ds.extend(get_prompt_item(name, profile, ds, meta_prompt, model_name, result_path))
        print(model_name, result_path)
    
        
def write_to_file(obj, dirname, lock):
    model_name = obj['model_name']
    output_path = os.path.join(dirname, f'interview_score_{model_name}_results_{aspect}.json')
    with lock:
        with open(output_path, 'a', encoding='utf-8') as fp:
            fp.write(json.dumps(obj, ensure_ascii=False, indent=2))
            
            
def post_process(obj):
    completions = obj['completions']
    answer = []
    for comp in completions:
        matches = re.findall(r'\s\d', comp)
        obj['matches'] = matches
        if len(matches):
            answer.append(matches[-1])
        else:
            answer.append(None)
    obj['answers'] = answer
    return obj
        

def api_worker(dataset, progress_bar, lock, post_fn, write_fn):
    global current_idx
    while True:
        with lock:
            idx = current_idx
            current_idx += 1
        if idx >= len(dataset):
            break
        obj = dataset[idx]
        prompt = obj['prompt']
        completion = decoder_for_openai('gpt-3.5-turbo', prompt, max_tokens=512, temperature=0.0, n=1, sys_prompt='You are a helpful and accurate assistant.')
        if isinstance(completion, str):
            completion = [completion]
        assert isinstance(completion, list), type(completion)
        obj['completions'] = completion
        obj = post_fn(obj)
        write_fn(obj)
        with lock:
            progress_bar.update()


threads = []
output_dir_with_aspect = os.path.join(output_dir, aspect)
os.makedirs(output_dir_with_aspect, exist_ok=True)
file_lock = Lock()
progress_lock = Lock()
gened_data = []
for fn in os.listdir(output_dir_with_aspect):
    if fn.endswith('.json'):
        gened_data.extend(read_gen_data(os.path.join(output_dir_with_aspect, fn)))
gened_keys = set()
for ex in gened_data:
    if ex['answer_path'] in gened_keys:
        print('dup', ex['answer_path'])
    gened_keys.add(ex['answer_path'])
new_prompt_ds = []
for ex in prompt_ds:
    if ex['answer_path'] in gened_keys:
        continue
    new_prompt_ds.append(ex)
print(f'total: {len(prompt_ds)}, new: {len(new_prompt_ds)}, completed: {len(gened_data)}')
prompt_ds = new_prompt_ds
write_fn = partial(write_to_file, dirname=output_dir_with_aspect, lock=file_lock)
progress_bar = tqdm(prompt_ds)
print(prompt_ds[0].keys())
# print(prompt_ds[-5]['prompt'])
# exit(0)

for i in range(n_workers):
    t = Thread(target=api_worker, args=(prompt_ds, progress_bar, progress_lock, post_process, write_fn))
    t.start()
    threads.append(t)
    
for t in threads:
    t.join()
