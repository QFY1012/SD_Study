from pathlib import Path
import json,csv,zipfile,hashlib,re
ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'report.json').read_text())
tasks=[json.loads(x) for x in (ROOT/'tasks_all.jsonl').read_text().splitlines()]
assert len(tasks)>=600 and len({t['id'] for t in tasks})==len(tasks)
# Keep the user's fixed anatomy tokens in every render prompt, outside embedding input.
base_prefix='@cheese, masterpiece, best quality, score_9, score_8, score_7, safe, 1boy, furry male, solo, nude'; style_suffix='cel shading, clean lineart, flat color, sharp shadows'
neg_prefix='worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, lowres, bad anatomy, extra digits, fused digits, digitigrade'
for t in tasks:
 allowed={'identity','cel_style','hand_count','finger_shape','foot_count','yellow_pads','human_pose'}
 removed=[v for v in t['targets'] if v not in allowed]
 t['targets']=[v for v in t['targets'] if v in allowed]
 t['scoring_targets']=[v for v in t['scoring_targets'] if v in allowed]
 if t['category']=='identity_control':
  t['targets']=[v for v in t['targets'] if v in ['identity','cel_style','human_pose']]
  t['scoring_targets']=[v for v in t['scoring_targets'] if v in ['identity','cel_style','human_pose']]
 if removed:t['normalization_notes']={'removed_non_rubric_labels':removed}
 task_prefix=', '.join([base_prefix]+(['blue fur'] if t['category'] in ('hand_detail','foot_detail') or 'close' in t['camera'].lower() else [])+(['yellow paw pads'] if 'yellow_pads' in t['scoring_targets'] else [])+[style_suffix]); inst=t['instruction'].strip(); inst=inst if not inst or inst[-1] in '.!?' else inst+'.'; t['prompt']=f"{task_prefix}, {inst} {t['camera'].strip().rstrip('.')}"; t['negative_prompt']=neg_prefix
for file,rows in [('tasks_all.jsonl',tasks),('train.jsonl',[t for t in tasks if t['split']=='train']),('eval.jsonl',[t for t in tasks if t['split']=='eval'])]:
 (ROOT/file).write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in rows))
with (ROOT/'tasks.csv').open('w',newline='') as f:
 fields=['id','split','category','action_family','intent_zh','instruction','camera','visible_hands','visible_soles','visibility_requirement','prompt','negative_prompt']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(tasks)
coordinates={r['id']:r for r in csv.DictReader((ROOT/'tsne.csv').open())}
points=[]
for t in tasks:
 r=coordinates[t['id']];points.append(dict(id=t['id'],x=float(r['x']),y=float(r['y']),category=t['category'],split=t['split'],text=t['intent_zh']+'\n'+t['instruction']))
data=json.dumps(points,ensure_ascii=False).replace('<','\\u003c')
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Anima 600 任务分布</title><style>body{font:16px system-ui;margin:24px;background:#f6f8fb;color:#172b42}canvas{background:white;border:1px solid #ccd5df;width:100%;max-width:1100px}select,input{padding:8px;margin-right:10px}pre{white-space:pre-wrap;background:white;padding:18px;min-height:90px;max-width:1064px}p{max-width:1100px}</style><h1>Anima 600 任务：文本 embedding 与 t-SNE</h1><p>悬停查看任务。圆点是训练任务，三角是独立提示词测试任务。二维距离仅供观察，去重依据原始 384 维向量；动作家族不保证跨集合隔离。</p><select id="cat"><option value="">全部类别</option></select><select id="split"><option value="">全部集合</option><option>train</option><option>eval</option></select><input id="query" placeholder="搜索任务文字或 ID"><p id="count"></p><canvas id="plot" width="1100" height="650"></canvas><pre id="tip">悬停或点击一个点查看任务。</pre><script id="data" type="application/json">DATA</script><script>
const points=JSON.parse(document.getElementById('data').textContent),cats=[...new Set(points.map(p=>p.category))],cv=document.getElementById('plot'),ctx=cv.getContext('2d'),palette=['#3366cc','#dc3912','#ff9900','#109618','#990099','#0099c6','#dd4477','#66aa00','#b82e2e','#316395','#994499','#22aa99'];let drawn=[];
for(const cat of cats){let o=document.createElement('option');o.value=cat;o.textContent=cat;document.getElementById('cat').append(o)}
const xmin=Math.min(...points.map(p=>p.x)),xmax=Math.max(...points.map(p=>p.x)),ymin=Math.min(...points.map(p=>p.y)),ymax=Math.max(...points.map(p=>p.y));
function draw(){ctx.clearRect(0,0,1100,650);drawn=[];let cat=document.getElementById('cat').value,split=document.getElementById('split').value,q=document.getElementById('query').value.toLowerCase();for(const p of points){if(cat&&p.category!==cat||split&&p.split!==split||q&&!(p.id+' '+p.text).toLowerCase().includes(q))continue;const x=30+(p.x-xmin)/(xmax-xmin)*1040,y=30+(p.y-ymin)/(ymax-ymin)*590;ctx.fillStyle=palette[cats.indexOf(p.category)];ctx.beginPath();if(p.split==='eval'){ctx.moveTo(x,y-5);ctx.lineTo(x-4,y+4);ctx.lineTo(x+4,y+4);ctx.closePath()}else ctx.arc(x,y,3.6,0,Math.PI*2);ctx.fill();drawn.push({p,x,y})}document.getElementById('count').textContent=drawn.length+' / '+points.length+' 个任务'}
function inspect(e){const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*1100/r.width,y=(e.clientY-r.top)*650/r.height;let closest=null,d=144;for(const v of drawn){const n=(v.x-x)**2+(v.y-y)**2;if(n<d){d=n;closest=v.p}}if(closest)document.getElementById('tip').textContent=closest.id+' | '+closest.category+' | '+closest.split+'\n'+closest.text}
cv.addEventListener('mousemove',inspect);cv.addEventListener('click',inspect);for(const id of ['cat','split','query'])document.getElementById(id).addEventListener('input',draw);draw();
</script></html>'''.replace('DATA',data)
(ROOT/'tsne.html').write_text(html)
usage={}
for p in (ROOT/'raw').glob('*.json'):
 u=json.loads(p.read_text()).get('usage') or {}
 for k in ['prompt_tokens','completion_tokens','total_tokens']:usage[k]=usage.get(k,0)+u.get(k,0)
report['generation_api_usage_successful_batches']=usage
report['usage_limitations']='Only cached successful generation responses are included; discarded attempts and reviews are not part of this usage subtotal.'
report['raw_generated_tasks']=sum(len(json.loads(p.read_text())['tasks']) for p in (ROOT/'raw').glob('*.json'))
report['review']='DeepSeek textual feasibility review; not human-certified and not a check of generated images.'
import importlib.metadata
def _pkg_ver(n,d):
 try:return importlib.metadata.version(n)
 except importlib.metadata.PackageNotFoundError:return d.get(n,'missing')
existing_pkgs=report.get('packages',{}) if isinstance(report.get('packages'),dict) else {}; report['packages']={name:_pkg_ver(name,existing_pkgs) for name in ['numpy','scikit-learn','sentence-transformers','torch']}
snapshots=list((ROOT/'model_cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots').glob('*'))
report['embedding_model_revision']=[p.name for p in snapshots]
report['content_sha256']=hashlib.sha256((ROOT/'tasks_all.jsonl').read_bytes()).hexdigest()
(ROOT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
readme=f'''# Anima 角色强化学习任务集 v0.1

总计 {len(tasks)} 条任务，480 条训练、120 条独立提示词评估，覆盖 12 类。各类数量见 report.json；类别配额以通过审查的任务为准，不强行凑齐每类数量。
生成模型 deepseek-flash，生成原始响应及用量见 raw/。未调用图片生成或 GPU。

## 语义去重

英文任务正文与镜头描述使用 sentence-transformers/all-MiniLM-L6-v2 编码为 384 维 L2 归一化向量。
相同角色、风格、五指/四趾等固定前缀不参与向量比较。先移除规范化文字重复，再以全局最远优先选择并以类别均衡、每类最多 60 条的方式选择。
任何保留任务对的余弦相似度必须低于 0.84。实际最大值 {report['maximum_pair_cosine']:.6f}；阈值以上任务对 {report['pairs_at_or_above_threshold']} 对。
embedding 是语义相似度代理，不能证明所有任务完全不重叠。nearest_pairs.csv 列出了最接近的 100 对，供人工复查。
t-SNE 仅作可视化，不能通过二维点是否重叠证明任务是否独立。

## 数据划分

每类留出 10 条 eval，其余为 train。ID 和完整提示词互不重复。评估固定种子为 101/202/303/404。
这是提示词留出，动作家族、身体部位或构图概念仍可能相似，不应宣称严格的动作家族隔离测试。可见性审查由 DeepSeek 对任务文本执行，审查结论不是人工认证。
原空间 train/eval 最大相似度为 {report['train_eval_max_cosine']:.6f}。
eval 不得进入 GRPO 提示词采样和奖励规则调参。

## 使用

- tasks_all.jsonl / tasks.csv：全部任务。
- train.jsonl / eval.jsonl：训练与评估集合。
- rubric.json：待人工校准的评分规则草案，不是已验证奖励模型。
- embeddings.npz：选中任务向量、原始候选向量和 ID。
- selected_indices.json：候选池选择索引；raw/ 文件按类别名称生成，候选顺序由 build.py 定义。
- tsne.html：独立交互页面，支持类别/集合/文字筛选和悬停查看。
- tsne.png / tsne.csv：静态图与二维坐标。
- report.json / nearest_pairs.csv：统计和近邻审查。

正式训练前应先用原 LoRA 生成基线图，由用户检查一批，再验证 VLM 对可见性、五指、四趾、黄色肉垫的评分准确率。
遮挡的手脚数量必须标记无法判断，不能默认正确；identity_control 不评分未要求可见的肉垫/手脚数量。
不把镜头关系相似或 t-SNE 图上靠近直接当作标签错误。
'''
(ROOT/'README.md').write_text(readme)
with zipfile.ZipFile(ROOT/'anima_tasks_600.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix in ['.jsonl','.csv','.json','.png','.html','.md','.npz','.py']:z.write(p,p.name)
 for folder in ['raw','review']:
  for p in (ROOT/folder).glob('*.json'):z.write(p,folder+'/'+p.name)
print('PACKAGE_READY',report)
