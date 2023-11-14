from eval_utils import LocalCharacter, PromptCharacter, get_character_names, PromptLocalCharacter, PromptInterviewer, seed_data_dir
from io_utils import read_jsonl, read_profile
import json
import os
from threading import Thread, Lock
from datetime import datetime
from tqdm import tqdm
import copy
from queue import Queue

question_path = './data/seed_data/questions/generated_agent_interview_for_multiturn_{name}.json'
output_dir = './evaluation_result'
dirname = 'interview_turns'
names = get_character_names()

n_workers = 8
now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
max_turns = 5
threads = []
import sys
model_type = sys.argv[1]
task_list = []
n_task_done = 0
DEBUG = False

if DEBUG:
    names = ['Beethoven']
else:
    names = get_character_names()


print('Characters:', names)
print('Model_type', model_type)

def worker(task_queue: Queue, **kwargs):
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.put(None)
            break
        interview_fn(**kwargs, item=item)
        

def interview_fn(lock: Lock, item: dict, pid: int):
    name = item['character_name']
    model_type = item['model_type']
    output_path = item['output_path']
    loc_time = 'Coffee Shop - Afternoon'
    status = f'{name} is casually chatting with a man from the 21st century. {name} fully trusts the man who engage in conversation and shares everything {name} knows without reservation.'
    if model_type != 'chatgpt':
        if model_type == 'sft':
            model_type = f"{name}-end"
        character = LocalCharacter(model_type, seed_data_dir, name=name, location=loc_time, status=status)
    else:
        character = PromptCharacter(seed_data_dir, name, location=loc_time, status=status)
    interviewer = PromptInterviewer(seed_data_dir, character.character_short_name, character.location, character.status)
    _, character_profile = read_profile(os.path.join(seed_data_dir, 'profiles', f'wiki_{character.character_short_name}.txt'))
    character_profile = character_profile[0]
    topic = item['question']
    interviewer.set_topic_and_profile(topic, character_profile)
    interviewer.start_conversation()
    character.start_conversation()
    results = {
        'character': character.character_name,
        'model': character.model_name,
        'topic': topic,
        'qid': item['qid'],
        'max_turns': max_turns,
        'finished': False,
    }
    print(json.dumps(results, ensure_ascii=False))
    content = []
    for turn_idx in tqdm(list(range(max_turns)), desc=character.character_short_name + '-' + character.model_name + f'-worker-{pid}', position=int(pid)):
        q = interviewer.get_reply()
        content.append({
            'turn_id': turn_idx,
            'turn_role': 'interviewer',
            'turn_content': q,
        })
        # print(q)
        interviewer.add_dialogue_history(q)
        character.add_dialogue_history(q[-1:])
        a = character.get_reply()
        for d in a:
            d['content'] = d['content'].split('\n\n')[0]
        
        content.append({
            'turn_id': turn_idx,
            'turn_role': 'character',
            'turn_content': a,
        })
        # print(a)
        interviewer.add_dialogue_history(a[-1:])
        character.add_dialogue_history(a)

    results['content'] = content
    results['finished'] = True
    with lock:
        with open(output_path, 'a', encoding='utf-8') as fp:
            print(json.dumps(results, ensure_ascii=False), file=fp)
    
if __name__ == '__main__':
    
    for name in names:
        loc_time = 'Coffee Shop - Afternoon'
        status = f'{name} is casually chatting with a man from the 21st century. {name} fully trusts the man who engage in conversation and shares everything {name} knows without reservation.'
        output_path = os.path.join(output_dir, dirname, f'multiturn_{name}_{model_type}_result/{now_str}.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(question_path.format(name=name), 'rb') as fp:
            questions = json.load(fp)
        if os.path.exists(output_path):
            gen_data = read_jsonl(output_path)
        else:
            gen_data = []
        task_done = {}
        for d in gen_data:
            if d['finished']:
                task_done[d['qid']] = True
        n_task_done += len(task_done)
        for idx, q in enumerate(questions):
            if idx in task_done:
                continue
            task_list.append({'qid': idx, 'question': q['question'], 'character_name': name, 'model_type': model_type, 'output_path': output_path})
            # if idx >= 10:
            #     break
    print('remain_interviews', len(task_list), 'done_interviews', n_task_done, 'total', len(task_list) + n_task_done)
    import random
    random.shuffle(task_list)
    task_queue = Queue()
    for t in task_list:
        task_queue.put(t)
    task_queue.put(None)
    print('task_queue size', task_queue.qsize())
    threads = []
    lock = Lock()
    for i in range(n_workers):
        t = Thread(target=worker, args=(task_queue,), kwargs={'lock': lock, 'pid': i})   
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
