import concurrent.futures as futures
import json,os,time,urllib.request,urllib.error,re,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CATEGORIES={
'one_hand':'The target palm faces the camera and touches no surface or prop; if another hand grips an object, only the free hand is counted. One hand clearly presenting its palm and all five long slender fingers; vary natural arm gestures and situations.',
'both_hands':'Both hands are FREE and touching NOTHING; palms face the camera. No holding props, praying palms together, gripping, palms pressed on hips, tables or walls. Vary body pose, arm height, shoulder orientation and camera angle while making both palm surfaces visible. Both palms and all five fingers on each hand clearly visible; varied human gestures, activities and compositions.',
'hand_detail':'Close and medium views of hands, finger separation and yellow pads, with meaningfully different gestures; avoid hiding required fingers.',
'one_sole':'One foot sole and all four toes clearly visible; varied natural human seated/standing/relaxed poses.',
'both_soles':'Both soles and all four toes per foot visible in comfortable human sitting or reclining poses.',
'foot_detail':'Foot and toe closeups, side and angled views with four distinct toes and visible yellow sole pads; varied contexts.',
'hand_and_foot':'At least one palm and one sole clearly visible together, showing five fingers and four toes respectively in anatomically plausible poses.',
'sitting':'Diverse natural human sitting poses and daily situations; at least one visible palm or sole, varied chairs/floor/supports.',
'reclining':'Diverse relaxed human reclining and lying poses; at least one visible palm or sole; no sexual poses.',
'moving':'Natural human walking, turning, stretching and gentle actions, plantigrade feet, at least one visible palm; no animal hock stance.',
'interaction':'Diverse everyday prop/environment interactions with one free open palm clearly visible; do not force occluded gripping digits to be counted.',
'identity_control':'Character identity and cel-shading preservation tests across scenes and compositions without requiring palm or sole visibility. No pad-count reward when hidden.'}
COMMON='@cheese, masterpiece, best quality, score_9, score_8, score_7, adult male anthropomorphic blue tiger, solo, blond hair, yellow eyes, blue stripes, pink star marking on abdomen, five fingers on each hand, long slender fingers, four toes on each foot, yellow paw pads, natural human-like legs, plantigrade feet, cel shading, clean lineart, flat colors, sharp shadows'
NEG='worst quality, low quality, score_1, score_2, score_3, blurry, bad anatomy, extra digits, fused digits'
SYSTEM='''Create a diverse benchmark of text-to-image tasks for an adult anthropomorphic blue tiger character. Output JSON only: {"tasks":[...]}. Each task must have instruction (English, 25-60 words, concrete renderable action and composition), action_family (short broad verb-based label, e.g. greeting, resting, balancing, reading; not unique IDs), camera (English shot description), targets (subset of ["identity","cel_style","hand_count","finger_shape","foot_count","yellow_pads","human_pose"]), visible_hands (0,1,2), visible_soles (0,1,2), visibility_requirement (English), intent_zh (Chinese, concise). No task IDs. No model jargon or rubric instructions inside instruction. Every task must be genuinely different in action/composition, not merely swapping clothing, colors, backgrounds, left/right or synonyms. Human-like leg anatomy, plantigrade feet. Five fingers per hand, four toes per foot, yellow pads on palms and soles. Fingers may be naturally bent but long/slender. For count targets the corresponding appendage and all digits must be clearly visible. Occluded appendages are unassessable, never presumed correct. No adult/sexual content. Keep character design unchanged. Do not add visible yellow paw pads as a blanket phrase. No repetitive standing-and-waving series. Keep at least one meaningful palm/sole target except identity_control. Identity/style always targets. Provide exactly 30 tasks.'''

def generate(cat,variant):
 out=ROOT/'raw'/f'{cat}_{variant}.json';out.parent.mkdir(exist_ok=True)
 if out.exists():return json.loads(out.read_text())
 key=os.environ['DEEPSEEK_API_KEY']
 body={'model':'deepseek-flash','messages':[{'role':'system','content':SYSTEM},{'role':'user','content':f'Category: {cat}. Scope: {CATEGORIES[cat]}\nBatch variant {variant}: '+('emphasize varied everyday actions and relaxed poses' if variant==0 else 'emphasize unusual but feasible actions, viewpoint and composition; avoid generic greeting, waving and resting examples')}], 'response_format':{'type':'json_object'},'max_tokens':14000,'temperature':1.0,'stream':False,'extra_body':{}}
 body.pop('extra_body');body['thinking']={'type':'disabled'}
 for attempt in range(3):
  try:
   req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=240) as r:result=json.load(r)
   content=result['choices'][0]['message']['content'];(ROOT/'raw'/f'debug_{cat}_{variant}.txt').write_text(content);data=json.loads(content);tasks=data['tasks']
   if not 20<=len(tasks)<=45:raise ValueError(f'Unexpected task count {len(tasks)}')
   for t in tasks:
    for k in ['instruction','action_family','camera','targets','visible_hands','visible_soles','visibility_requirement','intent_zh']:
     if k not in t:raise ValueError('Missing field '+k)
    assert isinstance(t['instruction'],str) and len(t['instruction'].split())>=15
    assert t['visible_hands'] in [0,1,2] and t['visible_soles'] in [0,1,2]
    if 'hand_count' in t['targets'] and not t['visible_hands']:raise ValueError('Hand count without visible hands')
    if 'foot_count' in t['targets'] and not t['visible_soles']:raise ValueError('Foot count without visible sole')
    t['category']=cat;t['generation_batch']=variant
   record={'tasks':tasks,'model':result.get('model'),'usage':result.get('usage'),'settings':{'temperature':1,'thinking':'disabled'}}
   out.write_text(json.dumps(record,ensure_ascii=False,indent=2));print('GENERATED',cat,variant,len(tasks),flush=True);return record
  except Exception as e:
   print('RETRY',cat,variant,attempt,type(e).__name__,str(e)[:160],flush=True)
   if attempt==2:raise
   time.sleep(2)

def review_category(cat,tasks):
 payload=[{'index':i,**{k:t[k] for k in ['instruction','camera','targets','visible_hands','visible_soles','visibility_requirement']}} for i,t in enumerate(tasks)]
 digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:12]
 dest=ROOT/'review'/f'{cat}_{digest}.json';dest.parent.mkdir(exist_ok=True)
 if dest.exists():decision=json.loads(dest.read_text())
 else:
  key=os.environ['DEEPSEEK_API_KEY']
  system='''Audit text-to-image tasks for physical feasibility and internal consistency. Return JSON {"approved_ids":[integer indices],"rejected":[{"index":integer,"reason":short English explanation}]}. Classify every input index exactly once. Be strict: a palm pressed against the floor or a wall is NOT visibly showing its pads unless a separate free hand is the declared target. Planted soles are hidden, side/top views cannot show the entire sole. Gloves cover palm pads; shoes/boots cover foot pads. Fingers gripping a prop may be occluded. A foot sole can face the camera only with a physically plausible supported/lifted pose. Standing with both soles facing camera is impossible without a support/transparent surface. Required five-finger or four-toe counts need all digits of the specified appendage visible. Requiring BOTH palms while one hand holds an object is inconsistent. If visibility requirement contradicts instruction or camera, reject; do not silently repair. Everyday harmless props and knees bent naturally in human poses are allowed. Identity-control tasks with no anatomy count targets need not show pads. Reject repetitive or nonrenderable abstract tasks too. Do not reject a task merely for sharing a broad body part with another task. This is textual auditing, not proof about future generated images.'''
  body={'model':'deepseek-flash','messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(payload)}],'response_format':{'type':'json_object'},'thinking':{'type':'disabled'},'temperature':0,'max_tokens':8000}
  req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=240) as r:result=json.load(r)
  decision=json.loads(result['choices'][0]['message']['content']);decision['model']=result.get('model');decision['usage']=result.get('usage');dest.write_text(json.dumps(decision,ensure_ascii=False,indent=2))
 ids=decision['approved_ids'];rejected=decision['rejected'];allids=ids+[r['index'] for r in rejected]
 if len(allids)!=len(tasks) or set(allids)!=set(range(len(tasks))):
  rejected_ids={r['index'] for r in rejected if isinstance(r.get('index'),int)}
  ids=sorted({i for i in ids if isinstance(i,int) and 0<=i<len(tasks)}-rejected_ids)
  decision['schema_warning']='Incomplete/duplicate classification; unclassified or conflicting items excluded conservatively.'
  decision['effective_approved_ids']=ids
  decision['unclassified_ids']=sorted(set(range(len(tasks)))-set(allids))
  dest.write_text(json.dumps(decision,ensure_ascii=False,indent=2))
 print('REVIEWED',cat,'approved',len(ids),'rejected',len(rejected),flush=True)
 return [tasks[i] for i in ids]

def main():
 with futures.ThreadPoolExecutor(max_workers=4) as pool:
  jobs=[pool.submit(generate,c,v) for c in CATEGORIES for v in range(3)]
  for job in futures.as_completed(jobs):job.result()
 print('GENERATION_COMPLETE',flush=True)
 approved={}
 for round_id in range(3,6):
  with futures.ThreadPoolExecutor(max_workers=4) as pool:
   jobs={cat:pool.submit(review_category,cat,[t for file in sorted((ROOT/'raw').glob(f'{cat}_*.json')) for t in json.loads(file.read_text())['tasks']]) for cat in CATEGORIES}
   approved={cat:job.result() for cat,job in jobs.items()}
  missing=[cat for cat,items in approved.items() if len(items)<20]
  if not missing:break
  with futures.ThreadPoolExecutor(max_workers=4) as pool:
   for job in futures.as_completed([pool.submit(generate,cat,round_id) for cat in missing]):job.result()
 else:raise RuntimeError('Not enough feasible candidate tasks')
 # Download public English sentence embedding model; no API secrets leave the DeepSeek request.
 import numpy as np
 import torch
 torch.set_num_threads(4)
 from sentence_transformers import SentenceTransformer
 model_name='sentence-transformers/all-MiniLM-L6-v2'
 model=SentenceTransformer(model_name,cache_folder=str(ROOT/'model_cache'),device='cpu')
 records=[];seen=set()
 for cat in CATEGORIES:
  for t in approved[cat]:
   normalized=re.sub(r'\W+',' ',t['instruction'].lower()).strip()
   if normalized in seen:continue
   seen.add(normalized);records.append(t)
 (ROOT/'approved_candidates.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
 # Remove the constant character/style prefix from embedding inputs: compare task-specific content.
 texts=[t['instruction']+' Camera: '+t['camera'] for t in records]
 emb=model.encode(texts,normalize_embeddings=True,batch_size=32,show_progress_bar=False)
 sim=emb@emb.T
 selected=[];quota={cat:0 for cat in CATEGORIES};threshold=.84
 # Global farthest-first selection with equal category quotas. Never silently relax threshold.
 maxsim=np.full(len(records),-1.,dtype=np.float32)
 for _ in range(600):
  eligible=np.array([i not in selected and quota[t['category']]<60 and maxsim[i]<threshold for i,t in enumerate(records)])
  if not eligible.any():raise RuntimeError(f'Only {len(selected)} tasks meet cosine threshold {threshold}; add candidates before claiming 600')
  # Fill the least represented categories first, then maximize semantic distance.
  minimum=min(quota[records[i]['category']] for i in np.where(eligible)[0])
  eligible &= np.array([quota[t['category']]==minimum for t in records])
  scores=np.where(eligible,maxsim,np.inf);idx=int(scores.argmin())
  selected.append(idx);quota[records[idx]['category']]+=1;maxsim=np.maximum(maxsim,sim[idx])
 chosen=[records[i].copy() for i in selected];vectors=emb[selected]
 # Evaluation is prompt-held-out, selected by semantic distance from its own seed per category.
 # This does not claim family-disjoint or zero semantic overlap across splits.
 train=[];evaluation=[]
 for cat in CATEGORIES:
  inds=[i for i,t in enumerate(chosen) if t['category']==cat]
  # Fixed deterministic split of the diverse selected tasks.
  rng=np.random.default_rng(42);rng.shuffle(inds)
  evaluation+=inds[:10];train+=inds[10:]
 evalset=set(evaluation)
 for i,t in enumerate(chosen):
  t['id']=f'cheese_rl_{i+1:04d}';t['split']='eval' if i in evalset else 'train'
  t['prompt']=COMMON+', '+t['instruction']
  t['negative_prompt']=NEG
  t['evaluation_seeds']=[101,202,303,404] if t['split']=='eval' else []
  t['expected']={'fingers_per_hand':5,'toes_per_foot':4,'pad_color':'yellow','occluded':'unassessable','human_leg_anatomy':True}
  t['scoring_targets']=list(dict.fromkeys(['identity','cel_style']+t['targets']))
  if t['split']=='eval':t['not_for_training']=True
 for name,items in [('tasks_all.jsonl',chosen),('train.jsonl',[chosen[i] for i in train]),('eval.jsonl',[chosen[i] for i in evaluation])]:
  (ROOT/name).write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in items))
 np.savez_compressed(ROOT/'embeddings.npz',embeddings=vectors,ids=np.array([t['id'] for t in chosen]),candidate_embeddings=emb)
 from sklearn.manifold import TSNE
 xy=TSNE(n_components=2,perplexity=30,random_state=42,init='pca',learning_rate='auto',metric='euclidean',max_iter=1500).fit_transform(vectors)
 import csv
 with (ROOT/'tsne.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['id','category','split','x','y'])
  for t,p in zip(chosen,xy):w.writerow([t['id'],t['category'],t['split'],*p])
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(13,9))
 colors=plt.get_cmap('tab20')
 for j,cat in enumerate(CATEGORIES):
  for split,marker in [('train','o'),('eval','^')]:
   inds=[i for i,t in enumerate(chosen) if t['category']==cat and t['split']==split]
   ax.scatter(xy[inds,0],xy[inds,1],s=22,alpha=.75,marker=marker,color=colors(j),label=cat if split=='train' else None)
 ax.set_title('600 Anima RL tasks | t-SNE of normalized text embeddings\nCircles: train (480), triangles: held-out prompts (120)')
 ax.legend(loc='center left',bbox_to_anchor=(1,0.5),fontsize=8);ax.set_xticks([]);ax.set_yticks([])
 fig.tight_layout();fig.savefig(ROOT/'tsne.png',dpi=180);plt.close(fig)
 sims=vectors@vectors.T;np.fill_diagonal(sims,-1)
 nearest=sims.max(1);tri=np.triu_indices(len(chosen),1)
 pairs=sorted(zip(sims[tri],tri[0],tri[1]),reverse=True)
 with (ROOT/'nearest_pairs.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['cosine_similarity','id_a','id_b','instruction_a','instruction_b'])
  for value,i,j in pairs[:100]:w.writerow([float(value),chosen[i]['id'],chosen[j]['id'],chosen[i]['instruction'],chosen[j]['instruction']])
 cross=sims[np.ix_(train,evaluation)]
 report={'tasks':len(chosen),'train':len(train),'eval':len(evaluation),'candidates':len(records),'category_counts':quota,'embedding_model':model_name,'embedding_dimensions':int(vectors.shape[1]),'embedding_input':'instruction + camera; invariant character/style prefix excluded','dedup_threshold':threshold,'maximum_pair_cosine':float(nearest.max()),'nearest_neighbor_median':float(np.median(nearest)),'train_eval_max_cosine':float(cross.max()),'pairs_at_or_above_threshold':int((sims[tri]>=threshold).sum()),'tsne':{'perplexity':30,'seed':42,'iterations':1500,'metric':'euclidean on L2-normalized embeddings'},'limitations':['Cosine threshold reduces near duplicates but does not guarantee all tasks are semantically distinct.','t-SNE is a visualization, not a de-duplication guarantee.','Split holds out prompts, not necessarily action families.','No image baseline or human/VLM rubric calibration has been performed.']}
 (ROOT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 # Preserve all generation records and selected candidate indices for reproducibility.
 (ROOT/'selected_indices.json').write_text(json.dumps(selected))
 print('RESULT',json.dumps(report),flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:traceback.print_exc();raise
