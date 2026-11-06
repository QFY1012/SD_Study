"""Runs ON the Colab VM. Queues SDXL (Illustrious-style) checkpoint + LoRA txt2img jobs on the local ComfyUI.

Reads /content/gen_sdxl_jobs.json: {"defaults": {...}, "jobs": [{"name": "...", "positive": "...", "seed": 1}]}
Per-job keys override defaults: negative, seed, width, height, steps, cfg, sampler, scheduler,
ckpt, lora, lora_strength_model, lora_strength_clip, prefix, clip_skip.
The LoRA file must be visible to ComfyUI (copy it into /content/ComfyUI/models/loras/ first).
Writes the prompt ids to /content/gen_sdxl_ids.json. Images land in <output_dir>/<prefix>/<name>_00001_.png.
"""
import json, urllib.error, urllib.request

DEFAULTS = dict(
    ckpt='xaviernocturne_v3.safetensors', lora='my_lora.safetensors',
    lora_strength_model=1.0, lora_strength_clip=1.0, clip_skip=None,
    width=768, height=1344, steps=28, cfg=5.5, sampler='euler_ancestral', scheduler='normal',
    negative='', seed=1, prefix='gen_sdxl')


def workflow(j):
    wf = {
        'ckpt': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': j['ckpt']}},
        'lora': {'class_type': 'LoraLoader', 'inputs': {
            'model': ['ckpt', 0], 'clip': ['ckpt', 1], 'lora_name': j['lora'],
            'strength_model': j['lora_strength_model'], 'strength_clip': j['lora_strength_clip']}},
        'pos': {'class_type': 'CLIPTextEncode', 'inputs': {'text': j['positive'], 'clip': ['lora', 1]}},
        'neg': {'class_type': 'CLIPTextEncode', 'inputs': {'text': j['negative'], 'clip': ['lora', 1]}},
        'lat': {'class_type': 'EmptyLatentImage', 'inputs': {'width': j['width'], 'height': j['height'], 'batch_size': 1}},
        'ks': {'class_type': 'KSampler', 'inputs': {
            'model': ['lora', 0], 'seed': j['seed'], 'steps': j['steps'], 'cfg': j['cfg'],
            'sampler_name': j['sampler'], 'scheduler': j['scheduler'], 'positive': ['pos', 0],
            'negative': ['neg', 0], 'latent_image': ['lat', 0], 'denoise': 1.0}},
        'dec': {'class_type': 'VAEDecode', 'inputs': {'samples': ['ks', 0], 'vae': ['ckpt', 2]}},
        'save': {'class_type': 'SaveImage', 'inputs': {'images': ['dec', 0], 'filename_prefix': j['prefix'] + '/' + j['name']}},
    }
    if j.get('clip_skip'):
        wf['skip'] = {'class_type': 'CLIPSetLastLayer', 'inputs': {'clip': ['lora', 1], 'stop_at_clip_layer': -int(j['clip_skip'])}}
        wf['pos']['inputs']['clip'] = ['skip', 0]
        wf['neg']['inputs']['clip'] = ['skip', 0]
    return wf


spec = json.load(open('/content/gen_sdxl_jobs.json'))
ids = []
for job in spec['jobs']:
    j = {**DEFAULTS, **spec.get('defaults', {}), **job}
    req = urllib.request.Request('http://127.0.0.1:8188/prompt', data=json.dumps({'prompt': workflow(j)}).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=30))
        ids.append({'name': j['name'], 'prompt_id': r['prompt_id']})
    except urllib.error.HTTPError as e:
        print('REJECTED', j['name'], e.code, e.read().decode()[:1500]); break
print('queued', len(ids), 'of', len(spec['jobs']))
json.dump(ids, open('/content/gen_sdxl_ids.json', 'w'))
