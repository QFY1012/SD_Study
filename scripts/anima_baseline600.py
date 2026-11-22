"""Incremental native Anima inference. One PNG and provenance JSON per task on Drive."""
import argparse,copy,gc,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);cli=parser.parse_args()
 c=json.loads(Path(cli.config).read_text());sys.path.insert(0,c['sd_scripts'])
 import torch
 from PIL import Image
 import anima_minimal_inference as infer
 from library import strategy_anima,strategy_base,anima_train_utils
 tasks=[json.loads(line) for line in Path(c['tasks']).read_text().splitlines()]
 assert len(tasks)==600 and len({t['id'] for t in tasks})==600
 for t in tasks:
  if not t['id'].replace('_','').isalnum():raise ValueError('Unsafe task id')
 for key in ['dit','text_encoder','vae','lora']:assert Path(c[key]).is_file(),c[key]
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'metadata').mkdir(exist_ok=True)
 provenance={k:v for k,v in c.items() if k not in ['sd_scripts','tasks']};provenance['tasks_sha256']=hashlib.sha256(Path(c['tasks']).read_bytes()).hexdigest();provenance['lora_sha256']=hashlib.sha256(Path(c['lora']).read_bytes()).hexdigest()
 fingerprint=digest(provenance)
 saved=out/'run_config.json'
 if saved.exists() and json.loads(saved.read_text())!=provenance:raise RuntimeError('Existing run configuration differs; refusing overwrite')
 saved.write_text(json.dumps(provenance,ensure_ascii=False,indent=2));shutil.copy2(c['tasks'],out/'tasks_all.jsonl')
 work=Path('/content/baseline600_work');work.mkdir(exist_ok=True)
 def atomic_json(path,obj):
  p=path.with_suffix('.tmp');p.write_text(json.dumps(obj,ensure_ascii=False,indent=2));os.replace(p,path)
 def complete(t):
  p=out/'images'/(t['id']+'.png');m=out/'metadata'/(t['id']+'.json')
  if p.exists() and m.exists():
   record=json.loads(m.read_text())
   if record['run_fingerprint']!=fingerprint or record['task_sha256']!=digest(t):raise RuntimeError('Existing task provenance differs: '+t['id'])
   with Image.open(p) as image:
    if image.size!=(c['width'],c['height']):raise RuntimeError('Existing image size mismatch')
    image.verify()
   return True
  return False
 done=[t for t in tasks if complete(t)];pending=[t for t in tasks if t not in done]
 atomic_json(out/'progress.json',{'total':600,'completed':len(done),'state':'loading','run_fingerprint':fingerprint})
 if not pending:return
 # Native parser supplies all upstream defaults; do not pass GRPO stochastic sampling here.
 original_argv=sys.argv
 sys.argv=['anima_minimal_inference.py','--dit',c['dit'],'--text_encoder',c['text_encoder'],'--vae',c['vae'],'--lora_weight',c['lora'],'--lora_multiplier','1','--save_path',str(work),'--image_size',str(c['height']),str(c['width']),'--infer_steps',str(c['steps']),'--guidance_scale',str(c['cfg']),'--flow_shift',str(c.get('flow_shift',5.0)),'--qwen_image_vae_2d','--attn_mode','torch','--device','cuda','--prompt','placeholder']
 try:args=infer.parse_args()
 finally:sys.argv=original_argv
 device=torch.device('cuda');settings=infer.get_generation_settings(args)
 model=infer.load_dit_model(args,device,torch.bfloat16)
 text_encoder=infer.load_text_encoder(args,device=device)
 vae=anima_train_utils.load_qwen_image_vae(args,device='cpu');vae.to(torch.bfloat16).eval()
 strategy_base.TokenizeStrategy.set_strategy(strategy_anima.AnimaTokenizeStrategy(qwen3_path=args.text_encoder,t5_tokenizer_path=None,qwen3_max_length=512,t5_max_length=512))
 strategy_base.TextEncodingStrategy.set_strategy(strategy_anima.AnimaTextEncodingStrategy())
 cache={};text_shared={'text_encoder':text_encoder,'conds_cache':cache};shared={'model':model}
 successes=len(done);started=time.time()
 for task in pending:
  task_args=copy.deepcopy(args);task_args.prompt=task['prompt'];task_args.negative_prompt=task['negative_prompt']
  task_args.seed=task['evaluation_seeds'][0] if task['evaluation_seeds'] else int(hashlib.sha256(task['id'].encode()).hexdigest()[:8],16)%(2**31)
  t0=time.time()
  try:
   with torch.no_grad():
    ctx,neg=infer.prepare_text_inputs(task_args,device,model,text_shared)
    latent=infer.generate(task_args,settings,shared,{'context':ctx,'context_null':neg})
    pixels=infer.decode_latent(vae,latent,device)
   pixels=((pixels.clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
   local=work/(task['id']+'.png');Image.fromarray(pixels).save(local)
   dest=out/'images'/local.name;temp=dest.with_suffix('.uploading.png');shutil.copyfile(local,temp);os.replace(temp,dest)
   atomic_json(out/'metadata'/(task['id']+'.json'),{'task_id':task['id'],'task_sha256':digest(task),'run_fingerprint':fingerprint,'prompt':task_args.prompt,'negative_prompt':task_args.negative_prompt,'seed':task_args.seed,'checkpoint':c['lora'],'seconds':time.time()-t0,'split':task['split'],'category':task['category']})
   local.unlink();successes+=1
   print('COMPLETED',task['id'],successes,'/600','seconds',round(time.time()-t0,2),flush=True)
   atomic_json(out/'progress.json',{'total':600,'completed':successes,'state':'running','last_task':task['id'],'elapsed_seconds':time.time()-started,'run_fingerprint':fingerprint})
   # Keep only the reused negative embedding. Each task's encoding is released after its image is saved.
   for key in list(cache):
    if key!=task_args.negative_prompt:del cache[key]
   del latent,pixels,ctx,neg
  except BaseException:
   atomic_json(out/'progress.json',{'total':600,'completed':successes,'state':'failed','failed_task':task['id'],'run_fingerprint':fingerprint});raise
 atomic_json(out/'progress.json',{'total':600,'completed':successes,'state':'complete','elapsed_seconds':time.time()-started,'run_fingerprint':fingerprint})
 print('ALL_600_COMPLETE',flush=True)

if __name__=='__main__':main()
