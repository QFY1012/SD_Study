from pathlib import Path
import json,re
import numpy as np
root=Path(__file__).resolve().parent
alltasks=[json.loads(x) for x in (root/'tasks_all.jsonl').read_text().splitlines()]
train=[json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()]
evalset=[json.loads(x) for x in (root/'eval.jsonl').read_text().splitlines()]
assert len(alltasks)==600 and len(train)==480 and len(evalset)==120
assert len({t['id'] for t in alltasks})==600
assert {t['id'] for t in train}.isdisjoint(t['id'] for t in evalset)
assert {t['prompt'] for t in train}.isdisjoint(t['prompt'] for t in evalset)
assert len({re.sub(r'\W+',' ',t['instruction'].lower()).strip() for t in alltasks})==600
allowed={'identity','cel_style','hand_count','finger_shape','foot_count','yellow_pads','human_pose'}
for t in alltasks:
 assert set(t['scoring_targets'])<=allowed
 assert 'identity' in t['scoring_targets'] and 'cel_style' in t['scoring_targets']
 if 'hand_count' in t['scoring_targets']:assert t['visible_hands']>0
 if 'foot_count' in t['scoring_targets']:assert t['visible_soles']>0
 if t['category']=='identity_control':assert not set(t['scoring_targets'])&{'hand_count','foot_count','yellow_pads','finger_shape'}
 assert t['camera'].strip().rstrip('.') in t['prompt'] and bool(t['visibility_requirement'].strip()) and t['visibility_requirement'] not in t['prompt'] and all(w not in t['prompt'].lower() for w in ('five fingers','four toes','plantigrade','off-frame'))
 assert t['expected']['fingers_per_hand']==5 and t['expected']['toes_per_foot']==4
 assert 'visible yellow paw pads' not in t['prompt'].lower()
a=np.load(root/'embeddings.npz');assert a['ids'].tolist()==[t['id'] for t in alltasks]
v=a['embeddings'];assert v.shape==(600,384)
np.testing.assert_allclose(np.linalg.norm(v,axis=1),1,atol=1e-5)
s=v@v.T;np.fill_diagonal(s,-1)
assert s.max()<.84
print(json.dumps({'validated':True,'total':600,'train':480,'eval':120,'maximum_cosine':float(s.max()),'threshold':.84,'exact_instruction_duplicates':0,'cross_split_prompt_duplicates':0,'scope':'Machine-checkable schema and embedding properties only; does not prove semantic or visual correctness.'},indent=2))
