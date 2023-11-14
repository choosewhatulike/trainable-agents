import os
import json
import re
import sys

data_path = sys.argv[1]
halluci_path = data_path.replace('dialogue', 'hallucination')
assert os.path.isfile(halluci_path), halluci_path

def read_profile(path):
    with open(path, 'r', encoding='utf-8') as fp:
        text = fp.read().strip()
    parts = text.split('\n\n')
    assert parts[0].startswith('# '), parts[0]
    agent_name = parts[0].replace('#', '').strip()
    agent_profile = []
    for p in parts[1:]:
        agent_profile.append(p.strip())
    return agent_name, agent_profile

agent_name = data_path.split('_')[-1].replace('.json', '')
seed_data_dir = './data/seed_data/'
character, _ = read_profile(os.path.join(seed_data_dir, 'profiles', f'wiki_{agent_name}.txt'))
out_path = os.path.join(os.path.dirname(data_path), 'prompted', f'prompted_agent_dialogue_{agent_name}.jsonl')

with open(os.path.join(seed_data_dir, 'prompts', 'agent_meta_prompt_sft.txt'), 'r', encoding='utf-8') as fp:
    meta_instruction = fp.read().strip()

with open(data_path, 'rb') as fp:
    data = json.load(fp)

sft_data = []
for ex in data:
    setting = ex['setting'][0]
    location = ex['location']
    prompt = meta_instruction.format(character=character, loc_time=location, status=setting)
    prompt += '\n\n'
    text = ''
    prev_role = ''
    prev_action = ''
    for turn in ex['dialogue']:
        role = turn['role']
        action = turn['action']
        if not action:
            action = '(speaking)'
        content = turn['content']
        if text and prev_role == role and prev_action == action:
            text += f'\n{content}'
        else:
            if text:
                text += '<|eot|>\n'
            prev_role = role
            prev_action = action
            text += f'{role} {action}: {content}'
    text += '<|eot|>'
    sft_data.append({
        'prompt': prompt,
        'output': text,
        'source': ex['source']
    })
    
# Hallucination Data
with open(halluci_path, 'rb') as fp:
    add_data = json.load(fp)
    
for ex in add_data:
    location = 'A room'
    setting = f'{agent_name} is chatting with a person.'
    prompt = meta_instruction.format(character=character, loc_time=location, status=setting)
    prompt += '\n\n'
    text = ''
    prev_role = ''
    prev_action = ''
    for turn in ex['dialogue']:
        role = turn['role']
        action = turn['action']
        action = '(speaking)'
        content = turn['content']
        if text and prev_role == role and prev_action == action:
            text += f'\n{content}'
        else:
            if text:
                text += '<|eot|>\n'
            prev_role = role
            prev_action = action
            text += f'{role} {action}: {content}'
    if not text:
        continue
    text += '<|eot|>'
    sft_data.append({
        'prompt': prompt,
        'output': text,
        'source': ex['source']
    })

print("output_path", out_path)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as fp:
    # json.dump(sft_data, fp, ensure_ascii=False, indent=2)
    for ex in sft_data:
        fp.write(json.dumps(ex, ensure_ascii=False) + '\n')

idx = 0
print('===SAMPLE INPUT===')
print(sft_data[idx]['prompt'])
print('===SAMPLE TARGET===')
print(sft_data[idx]['output'])
print(len(data), len(add_data),len(sft_data))
