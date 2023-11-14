import requests
import msgpack
import os.path as osp
import os
import re
import json
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    stop_never,
    wait_random_exponential,
)  # for exponential backoff
from api_call_util import decoder_for_openai, decoder_for_local_completion, decoder_for_local_chat, set_proxy, unset_proxy
import apikeys
API_KEY = apikeys.apikey_list[0]

seed_data_dir = './data/seed_data/'
 
def get_character_names():
    return [
        'Caesar',
        'Spartacus',
        'Voldemort',
        'Newton',
        'Socrates',
        'Beethoven',
        'Cleopatra',
        'Hermione',
        'Martin'
    ]


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

def get_output(res):
    out = res['message']['content']
    if res['finish_reason'] != 'stop':
        out += '<|NONSTOP|>'
    return out


class Character:
    def __init__(self, seed_data_dir, name, location=None, status=None) -> None:
        self.seed_data_dir = seed_data_dir,
        self.character_short_name = name
        self.model_name = 'pjeval'
        self.location = location
        self.status = status
        self.backend = 'sft'
        
        with open(osp.join(seed_data_dir, 'prompts', 'meta_prompt_agent_dialogue.txt'), 'r', encoding='utf-8') as fp:
            self.meta_instruction = fp.read().strip()
        self.character_name, _ = read_profile(osp.join(seed_data_dir, 'profiles', f'wiki_{name}.txt'))
        self.dialogue_history = []
        
    def set_scene(self, location, status):
        self.location = location.strip()
        self.status = status.strip()
        
    def start_conversation(self):
        self.dialogue_history.clear()
    
    def add_dialogue_history(self, dialogue):
        self.dialogue_history.extend(dialogue)
        
    def get_prompt(self):
        prompt = self.meta_instruction.format(
            character=self.character_name, loc_time=self.location, status=self.status
        )
        prompt += '\n\n'
        text = ''
        for d in self.dialogue_history[-8:]:
            role = d['role']
            action = d['action']
            content = d['content']
            if role == self.character_short_name or action == '(speaking)':
                text += f'{role} {action}: {content}<|eot|>\n'
        text += f'{self.character_short_name} ('
        return prompt + text
    
    def post_process(self, text: str):
        text = f'{self.character_short_name} (' + text
        # print('output:', text)
        sp_pos = text.find(f'(speaking):')
        if sp_pos != -1:
            pos = text.find('<|eot|>', sp_pos)
            if pos != -1:
                text = text[:pos]
                
        dialogue = []
        for line in text.split('<|eot|>'):
            line = line.strip()
            if not line:
                continue
            if ': ' not in line:
                part = ''
                content = line
            else:
                part = line.split(': ')[0]
                content = '\n\n'.join(line.split(': ')[1:])
            action = re.findall(r'\(.*?\)', part)[0]
            role = part.replace(action, '').strip()
            if len(role) == 0:
                role = self.character_short_name
            dialogue.append({
                'role': role,
                'action': action,
                'content': content
            })
        # print('dialogue:', dialogue)
        return dialogue
    
    @retry(stop=stop_after_attempt(3))
    def get_reply(self):
        payload = {
            'prompt': self.get_prompt(),
            'max_new_tokens': 256,
            'temperature': 0.2,
        }
        response = requests.post(self.url, data=msgpack.packb(payload))
        # print(payload['prompt'])
        # print(response.status_code)
        dialogue = self.post_process(response.text)
        return dialogue
    
class LocalCharacter(Character):
    def __init__(self, model_name, seed_data_dir, name, location=None, status=None) -> None:
        super().__init__(seed_data_dir, name, location, status)
        self.model_name = model_name
    
    @retry(stop=stop_after_attempt(3))
    def get_reply(self):
        max_length = 256
        temperature = 0.2
        unset_proxy()
        response = openai.Completion.create(
            api_key="EMPTY",
            api_base="http://localhost:8000/v1",
            model=self.model_name,
            prompt=self.get_prompt(),
            max_tokens=max_length,
            temperature=temperature,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            stop='<|eot|>',
        )
        # print(payload['prompt'])
        def _get_output(res):
            out = res['text']
            if res['finish_reason'] != 'stop':
                out += '<|NONSTOP|>'
            return out
        content = _get_output(response["choices"][0])
        dialogue = self.post_process(content)
        return dialogue

class PromptCharacter(Character):
    def __init__(self, seed_data_dir, name, location=None, status=None) -> None:
        self.seed_data_dir = seed_data_dir,
        self.character_short_name = name
        self.location = location
        self.status = status
        self.backend = 'chatgpt'
        self.model_name = 'gpt-3.5-turbo'
        
        with open(osp.join(seed_data_dir, 'prompts', 'agent_meta_prompt_chatgpt.txt'), 'r', encoding='utf-8') as fp:
            self.meta_instruction = fp.read().strip()
        self.character_name, _ = read_profile(osp.join(seed_data_dir, 'profiles', f'wiki_{name}.txt'))
        self.dialogue_history = []
        
    def get_prompt(self):
        prompt = self.meta_instruction.format(
            character=self.character_name, loc_time=self.location, status=self.status
        )
        prompt += '\n\n'
        text = ''
        for d in self.dialogue_history[-8:]:
            role = d['role']
            action = d['action']
            content = d['content']
            if role == self.character_short_name or action == '(speaking)':
                text += f'{role} {action}: {content}\n\n'
        text += f'{self.character_short_name} ('
        return prompt + text
        
    def post_process(self, text: str):
        text = f'{self.character_short_name} ' + text
        # print('output:', text)
        sp_pos = text.find(f'(speaking):')
        if sp_pos != -1:
            pos = text.find('\n\n', sp_pos)
            if pos != -1:
                text = text[:pos]
                
        dialogue = []
        for line in text.split('\n\n'):
            line = line.strip()
            if not line:
                continue
            if ': ' not in line:
                part = ''
                content = line
            else:
                part = line.split(': ')[0]
                content = '\n\n'.join(line.split(': ')[1:])
            action = re.findall(r'\(.*?\)', part)
            if len(action) == 0:
                action = '(speaking)'
                role = self.character_short_name
            else:
                action = action[0]
                role = part.replace(action, '').strip()
                if len(role) == 0:
                    role = self.character_short_name
            dialogue.append({
                'role': role,
                'action': action,
                'content': content
            })
        # print('dialogue:', dialogue)
        return dialogue
        
    @retry(stop=stop_after_attempt(3))
    def get_reply(self):
        max_length = 256
        temperature = 0.2
        prompt = self.get_prompt()
        response = decoder_for_openai('gpt-3.5-turbo', prompt, max_length, temperature, apikey=API_KEY, stop='\n\n')
        # print('[START]\n' + prompt + '\n[END]')
        dialogue = self.post_process(response)
        return dialogue
    
class PromptInterviewer(PromptCharacter):
    def __init__(self, seed_data_dir, name, location=None, status=None) -> None:
        self.seed_data_dir = seed_data_dir,
        self.character_short_name = name
        self.location = location
        self.status = status
        self.backend = 'chatgpt_interviewer'
        self.role = 'Man'
        
        with open(osp.join(seed_data_dir, 'prompts', 'agent_meta_prompt_interviewer_chatgpt.txt'), 'r', encoding='utf-8') as fp:
            self.meta_instruction = fp.read().strip()
        self.character_name, profiles = read_profile(osp.join(seed_data_dir, 'profiles', f'wiki_{name}.txt'))
        self.dialogue_history = []
        self.current_topic = None
        self.current_profile = None
        
    def set_topic_and_profile(self, topic: str, profile: str):
        self.current_topic = topic
        self.current_profile = profile
        
    def get_prompt(self):
        assert self.current_topic, self.current_topic
        assert self.current_profile, self.current_profile
        prompt = self.meta_instruction.format(
            character=self.character_name, loc_time=self.location, status=self.status,
            topic=self.current_topic, profile=self.current_profile,
        )
        prompt += '\n\n'
        text = ''
        for d in self.dialogue_history:
            role = d['role']
            action = d['action']
            content = d['content']
            if role == self.character_short_name or action == '(speaking)':
                text += f'{role} {action}: {content}\n\n'
        text += f'{self.role} (speaking):'
        return prompt + text
    
    def post_process(self, text: str):
        # print('output:', text)
        re.sub(r'\(.*?\)', '', text)
        content = text.split('\n\n')[0]
        dialogue = []
        dialogue.append({
            'role': self.role,
            'action': '(speaking)',
            'content': content.strip()
        })
        return dialogue
    
    def get_reply(self):
        max_length = 128
        temperature = 0.2
        prompt = self.get_prompt()
        response = decoder_for_openai('gpt-3.5-turbo', prompt, max_length, temperature, apikey=API_KEY)
        dialogue = self.post_process(response)
        return dialogue
    
    
class PromptLocalCharacter(PromptCharacter):
    def __init__(self, model_name, seed_data_dir, name, location=None, status=None) -> None:
        super().__init__(seed_data_dir, name, location, status)
        self.model_name = model_name
        self.backend = model_name
        
    def get_prompt(self):
        prompt = self.meta_instruction.format(
            character=self.character_name, loc_time=self.location, status=self.status
        )
        prompt += '\n\n'
        text = ''
        for d in self.dialogue_history[-8:]:
            role = d['role']
            action = d['action']
            content = d['content']
            if role == self.character_short_name or action == '(speaking)':
                text += f'{role} {action}: {content}\n\n'
        text += f'{self.character_short_name} (speaking): '
        return prompt + text

    def get_reply(self):
        max_length = 256
        temperature = 0.2
        prompt = self.get_prompt()
        response = decoder_for_openai('gpt-3.5-turbo', self.model_name, prompt, max_length, temperature, apikey=API_KEY)
        # print('[START]\n' + prompt + '\n[END]')
        # print(response)
        # dialogue = self.post_process(response)
        # print(dialogue)
        dialogue = {
            'role': self.character_short_name,
            'action': '(speaking)',
            'content': response,
        }
        return dialogue

