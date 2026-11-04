"""Runs ON the Colab VM. Queues Anima + LoRA txt2img jobs on the local ComfyUI.

Reads /content/gen_jobs.json: {"defaults": {...}, "jobs": [{"name": "...", "positive": "...", ...}]}
Per-job keys override defaults: negative, seed, width, height, steps, cfg, sampler, scheduler,
unet, clip, vae, lora, lora_strength, prefix.
Writes the prompt ids to /content/gen_ids.json. Images land in <output_dir>/<prefix>/<name>_00001_.png.
"""
import json, os, urllib.error, urllib.request

DEFAULTS = dict(
    unet='waiANIMA_v10Base10.safetensors', clip='waiANIMA_v10Base10_txt.safetensors',
    vae='qwenImage_qwenImageVAE.safetensors', lora='cheesev1.safetensors', lora_strength=1.0,
    width=768, height=1344, steps=30, cfg=4.5, sampler='er_sde', scheduler='simple',
    negative='', seed=1, prefix='gen')


def workflow(j):
    wf = {
        'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': j['unet'], 'weight_dtype': 'default'}},
        'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': j['clip'], 'type': j.get('clip_type', 'stable_diffusion'), 'device': 'default'}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': j['vae']}},
        'pos': {'class_type': 'CLIPTextEncode', 'inputs': {'text': j['positive'], 'clip': ['clip', 0]}},
        'neg': {'class_type': 'CLIPTextEncode', 'inputs': {'text': j['negative'], 'clip': ['clip', 0]}},
        'lat': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': j['width'], 'height': j['height'], 'batch_size': 1}},
        'ks': {'class_type': 'KSampler', 'inputs': {
            'model': ['unet', 0], 'seed': j['seed'], 'steps': j['steps'], 'cfg': j['cfg'],
            'sampler_name': j['sampler'], 'scheduler': j['scheduler'], 'positive': ['pos', 0],
            'negative': ['neg', 0], 'latent_image': ['lat', 0], 'denoise': 1.0}},
        'dec': {'class_type': 'VAEDecode', 'inputs': {'samples': ['ks', 0], 'vae': ['vae', 0]}},
        'save': {'class_type': 'SaveImage', 'inputs': {'images': ['dec', 0], 'filename_prefix': j['prefix'] + '/' + j['name']}},
    }
    if j.get('lora'):
        wf['lora'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['unet', 0], 'lora_name': j['lora'], 'strength_model': j['lora_strength']}}
        wf['ks']['inputs']['model'] = ['lora', 0]
    return wf


spec = json.load(open('/content/gen_jobs.json'))
ids = []
for job in spec['jobs']:
    j = {**DEFAULTS, **spec.get('defaults', {}), **job}
    req = urllib.request.Request('http://127.0.0.1:8188/prompt', data=json.dumps({'prompt': workflow(j)}).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=30))
        ids.append({'name': j['name'], 'prompt_id': r['prompt_id']})
        print('queued', j['name'], r['prompt_id'][:8])
    except urllib.error.HTTPError as e:
        print('REJECTED', j['name'], e.code, e.read().decode()[:1500]); break
json.dump(ids, open('/content/gen_ids.json', 'w'))
