import os
import sys
import json

path = sys.argv[1]
with open(path, 'r') as fp:
    results = json.load(fp)
    
path2 = path.replace('_dialogue_', '_hallucination_')
with open(path2, 'r') as fp:
    results.extend(json.load(fp))

sum_turns = 0
sum_turn_words = 0
for r in results:
    sum_turns += len(r['dialogue'])
    for d in r['dialogue']:
        sum_turn_words += len(d['content'].split())
print(f'total scenes: {len(results)}, total turn:{sum_turns}, avg turn:{sum_turns/len(results)}, total words: {sum_turn_words}, avg words: {sum_turn_words/sum_turns}')
