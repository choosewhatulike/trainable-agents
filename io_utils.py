import json
import os
import random

Dataset_Folder = './data/seed_data'

def read_gen_data(path):
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


def read_json(path):
    with open(path, 'rb') as fp:
        return json.load(fp)


def read_jsonl(path):
    results = []
    with open(path, 'rb') as fp:
        for line in fp:
            if line.strip():
                results.append(json.loads(line))
    return results


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        for ex in data:
            print(json.dumps(ex, ensure_ascii=False), file=fp)
            
            
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

def read_file(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return fp.read().strip()
    
def load_seed_data_train(args):
    questions = []
    # read dataset file and profiles
    random.seed(42)
    agent_short_name = args.character
    dataset_name = args.prompt_name
    prompt_dir = os.path.join(Dataset_Folder, 'prompts')
    profile_dir = os.path.join(Dataset_Folder, 'profiles')
    profile_path = os.path.join(profile_dir, f'wiki_{agent_short_name}.txt')
    agent_name, agent_profile = read_profile(profile_path)
    if dataset_name == 'gen_scene':
        max_seed_prompts = 100
        prompt_path = os.path.join(prompt_dir, 'prompt_agent_scene.txt')
        prompt = read_file(prompt_path)
        dup_times = int(max_seed_prompts / len(agent_profile))
        dup_times = max(dup_times, 1)
        for p in agent_profile:
            questions.append(prompt.format(agent_summary=p, agent_name=agent_name))
        new_questions = []
        for i in range(dup_times):
            new_questions.extend(questions)
        if len(new_questions) < max_seed_prompts:
            probs = [len(q) for q in questions]
            new_questions.extend(random.choices(questions, weights=probs, k=max_seed_prompts - len(new_questions)))
        questions = new_questions
    elif dataset_name == 'gen_dialogue':
        prompt_path = os.path.join(prompt_dir, 'prompt_agent_dialogue.txt')
        prompt = read_file(prompt_path)
        scene_path = os.path.join(args.data_path, f'generated_agent_scene_{agent_short_name}.json')
        with open(scene_path, 'r') as fp:
            scene_data = json.load(fp)
        for scene in scene_data:
            questions.append(prompt.format(
                agent_name=agent_name, agent_short_name=agent_short_name,
                agent_summary=scene['profile'],
                type=scene['type'], location=scene['location'], background=scene['background']))
    elif dataset_name == 'gen_hallucination':
        prompt_path = os.path.join(prompt_dir, 'prompt_agent_dialogue_adv.txt')
        prompt = read_file(prompt_path)
        for p in agent_profile:
            questions.append(prompt.format(
                agent_name=agent_name, agent_short_name=agent_short_name,
                agent_summary=p))
            
    prompt_ds = []
    for idx, q in enumerate(questions):
        prompt_ds.append({
            'prompt': q,
            'gen_answer_id': idx,
        })
    return prompt_ds
    