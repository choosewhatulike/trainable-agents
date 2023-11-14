from eval_utils import LocalCharacter, PromptCharacter, get_character_names, PromptLocalCharacter, seed_data_dir
import json
import os
from threading import Thread
from datetime import datetime
from tqdm import tqdm
import copy

question_path = './data/seed_data/questions/generated_agent_interview_{name}.json'
output_dir = './evaluation_result'
dirname = 'interview_single'
DEBUG = False

if DEBUG:
    model_lists = ['chatgpt']
    names = ['Beethoven']
else:
    model_lists = ['sft', 'alpaca-7b', 'chatgpt', 'vicuna-7b']
    names = get_character_names()

print('Characters:', names)
print('Model_list:', model_lists)
now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

def interview_worker(character, output_path):
    outputs = []
    for q in tqdm(questions, desc=character.character_short_name + '-' + character.model_name):
        character.start_conversation()
        q_text = q['question']
        character.add_dialogue_history([{
            'role': role,
            'action': '(speaking)',
            'content': q_text,
        }])
        d = character.get_reply()
        while character.backend == 'sft' and 'speaking' not in d[-1]['action']:
            character.add_dialogue_history(d)
            d = character.get_reply()
        outputs.append({
            'topic_id': q['topic_id'],
            'question': q_text,
            'reply': d, 
            })
        
    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(outputs, fp, ensure_ascii=False, indent=2)


threads = []

for name in names:
    loc_time = 'Coffee Shop - Afternoon'
    status = f'{name} is casually chatting with a man from the 21st century. {name} fully trusts the man who engage in conversation and shares everything {name} knows without reservation.'
    role = 'Man'
    with open(question_path.format(name=name), 'rb') as fp:
        questions = json.load(fp)
    questions = questions
    print(len(questions))
    for model_type in model_lists:
        if model_type == 'sft':
            character = LocalCharacter(model_type, seed_data_dir, name, location=loc_time, status=status)
            output_path = os.path.join(output_dir, dirname, f'{name}_{model_type}_result/{now_str}.json')
        elif model_type == 'chatgpt':
            character = PromptCharacter(seed_data_dir, name, location=loc_time, status=status)
            output_path = os.path.join(output_dir, dirname, f'{name}_chatgpt_result/{now_str}.json')
        elif model_type.endswith('-7b') or model_type.endswith('-13b') or model_type.endswith('-hf'):
            character = PromptLocalCharacter(model_type, seed_data_dir, name, location=loc_time, status=status)
            output_path = os.path.join(output_dir, dirname, f'{name}_{model_type}_result/{now_str}.json')
        else:
            raise NotImplementedError
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            print(f'{output_path} already exists! Skipping...')
            continue
        t = Thread(target=interview_worker, args=(copy.deepcopy(character), output_path))
        t.start()
        threads.append(t)
   
for t in threads:
    t.join()
