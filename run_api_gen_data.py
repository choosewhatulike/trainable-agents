from api_call_util import decoder_for_openai
from io_utils import read_json, read_gen_data, read_jsonl, load_seed_data_train
import os
import glob
from threading import Thread, Lock
import json
from functools import partial
import re
from tqdm import tqdm
import sys
from apikeys import apikey_list
from config import args

now = '2023-10-08'
current_idx = 0
prompt_dir = ''
temperature = args.temperature
model_name = args.model_name
prompt_name = args.prompt_name
output_path = f'./result/{now}/{prompt_name}/{model_name}-temp-{temperature}-char-{args.character}.jsonl'
n_workers = 16

# chatgpt_system_prompt = f"""You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. Knowledge cutoff: 2021-09 Current date: {now}"""
chatgpt_system_prompt = None


def check_result(text):
    return len(text) > 0

prompt_ds = load_seed_data_train(args)

def write_to_file(obj, output_path, lock):
    with lock:
        with open(output_path, 'a', encoding='utf-8') as fp:
            fp.write(json.dumps(obj, ensure_ascii=False, indent=2))


def api_worker(dataset, progress_bar, lock, write_fn, apikey):
    global current_idx
    cur_task_done_retry = 0
    while True:
        if cur_task_done_retry <= 0:
            with lock:
                idx = current_idx
                current_idx += 1
            if idx >= len(dataset):
                break
        obj = dataset[idx]
        prompt = obj['prompt']
        completion = ''
        try:
            completion = decoder_for_openai(model_name, prompt, max_tokens=args.max_tokens, temperature=temperature, n=1, sys_prompt=chatgpt_system_prompt, apikey=apikey)
        except Exception as e:
            print(repr(e))
            cur_task_done_retry = 100
        assert isinstance(completion, str), type(completion)
        obj['completions'] = completion
        res = check_result(completion)
        obj['check_result'] = res
        if not res:
            cur_task_done_retry += 1
            if cur_task_done_retry > 3:
                obj['retry_time'] = cur_task_done_retry
                write_fn(obj)
                print(f'failed for index {idx}')
                with lock:
                    progress_bar.update()
                cur_task_done_retry = 0
            continue
        else:
            obj['retry_time'] = cur_task_done_retry + 1
            cur_task_done_retry = 0
            write_fn(obj)
            with lock:
                progress_bar.update()


threads = []
output_path = os.path.abspath(output_path)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
file_lock = Lock()
progress_lock = Lock()

# check input_ds
answer_keys = {}
count = 0
for ex in prompt_ds:
    id_str = ex['gen_answer_id']
    if id_str in answer_keys:
        print('dup', id_str)
        count += 1
    else:
        answer_keys[id_str] = 1

if count > 0:
    print(f'total: {len(prompt_ds)}, repeated: {count}, please check gen_answer_id and remove duplicate, exiting...')
    exit(0)

gened_data = []
if os.path.exists(output_path):
    gened_data.extend(read_gen_data(output_path))
gened_keys = set()
for ex in gened_data:
    # if not ex['completions']:
    #     continue
    if not ex['check_result']:
        continue
    if ex['gen_answer_id'] in gened_keys:
        print('dup', ex['gen_answer_id'])
    gened_keys.add(ex['gen_answer_id'])
new_prompt_ds = []
for ex in prompt_ds:
    if ex['gen_answer_id'] in gened_keys:
        continue
    new_prompt_ds.append(ex)
print(f'total: {len(prompt_ds)}, new: {len(new_prompt_ds)}, completed: {len(gened_keys)}')
prompt_ds = new_prompt_ds
write_fn = partial(write_to_file, output_path=output_path, lock=file_lock)
if len(prompt_ds) == 0:
    print('Finished!')
    exit(0)
if args.debug:
    print(prompt_ds[0].keys())
    print(prompt_ds[0]['prompt'])
    prompt_ds = prompt_ds[:10]
progress_bar = tqdm(prompt_ds)


assert len(apikey_list) >= 1, f"need at least one apikeys, find {len(apikey_list)}"
for i in range(n_workers):
    api_idx = i % len(apikey_list)
    t = Thread(target=api_worker, args=(prompt_ds, progress_bar, progress_lock, write_fn, apikey_list[api_idx]))
    threads.append(t)
    
for t in threads:
    t.start()

for t in threads:
    t.join()
