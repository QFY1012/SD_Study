"""Runs ON the Colab VM. Local (box) repaint of wrongly drawn hands / feet with the SDXL checkpoint.

Port of the user's repair_regions.py: same detailer settings, same hand/feet prompts, optional
`restore_original`, but fully remote (no local ComfyUI, no run_batch, no Impact Pack).

Reads /content/repair_jobs.json:
{"ckpt": "xaviernocturne_v3.safetensors",            # optional
 "jobs": [{
    "name": "t01_soccer",
    "source": "/content/test_out/cheese_sports/t01_soccer_00001_.png",   # path on the VM
    "restore_original": "/content/....png",           # optional: reset boxed areas to this image first
    "stages": [                                       # chained: each stage repaints the previous result
       {"kind": "hands" | "feet" | "feet_strong", "boxes": [[x, y, w, h], ...]},
       {"kind": "feet",  "boxes": [[x, y, w, h], ...], "positive_suffix": "...", "negative_suffix": "...",
        "positive_text": "...", "denoise": 0.4, "seed": 123}
 ]}]}
Writes prompt ids to /content/repair_ids.json; results: /content/test_out/repair/<name>_*.png
"""
import json, os, secrets, shutil, urllib.error, urllib.request

INPUT_DIR = os.environ.get('COMFY_INPUT', '/content/test_input')
CKPT_DEFAULT = 'xaviernocturne_v3.safetensors'

# Prompts are identical to the user's repair_regions.py / run workflow.
#  - hands: node-8/9 text of the template.
#  - feet : repair_regions.py derives it from the hands text with the same three replace() calls.
#  - feet_strong: the refined feet prompt used in the two-stage workflow PNGs (four toes:1.3 ...).
# Reference anatomy: hands = 5 fingers, yellow paw pads; feet = 4 toes, claws, yellow pads.
HANDS_POS = 'masterpiece, best quality, absurdres, very aesthetic, crotch tuft, grey blue fur, (five fingers:1.1), light yellow paw pads, paw'
BASE_NEG = 'blurry, malformed anatomy, extra digits, missing fingers, extra limbs, distorted face, deformed eyes, nsfw'
PRESETS = {
    'hands': dict(positive=HANDS_POS, negative=BASE_NEG),
    'feet': dict(
        positive=HANDS_POS.replace('(five fingers:1.1)', '(four toes:1.1)').replace(', light yellow paw pads', '').replace(
            ', paw', ', digitigrade feet, clawed toes, original foot orientation'),
        negative=BASE_NEG),
    'feet_strong': dict(
        positive='masterpiece, best quality, absurdres, very aesthetic, crotch tuft, grey blue fur, (four toes:1.3), light yellow toe pads, yellow claws, digitigrade feet, clawed toes, original foot orientation',
        negative=BASE_NEG + ', human feet, hands, fingers, five toes, six toes, pink paw pads, dewclaws, extra ankle claws'),
}
DETAILER = dict(guide_size=512.0, guide_size_for=True, max_size=1024.0, steps=15, cfg=3.0,
                sampler_name='euler', scheduler='simple', denoise=0.4, feather=30, noise_mask=True,
                force_inpaint=True, cycle=1, noise_mask_feather=30, tiled_encode=False, tiled_decode=False,
                loras='[]', url='', token='')


def build(job, ckpt):
    name = job['name']
    g = {'ckpt': {'class_type': 'CheckpointLoaderSimple', 'inputs': {'ckpt_name': ckpt}}}
    src_name = f'{name}-source.png'
    shutil.copy2(job['source'], os.path.join(INPUT_DIR, src_name))
    g['load'] = {'class_type': 'LoadImage', 'inputs': {'image': src_name}}
    g['size'] = {'class_type': 'GetImageSize', 'inputs': {'image': ['load', 0]}}
    prev = ['load', 0]
    for i, st in enumerate(job['stages']):
        kind = st['kind']
        preset = PRESETS[kind]
        boxes = [dict(zip(('x', 'y', 'width', 'height'), b)) for b in st['boxes']]
        pos = st.get('positive_text') or preset['positive']
        if st.get('positive_suffix'):
            pos += ', ' + st['positive_suffix']
        neg = preset['negative'] + (', ' + st['negative_suffix'] if st.get('negative_suffix') else '')
        g[f'box{i}'] = {'class_type': 'CreateBoundingBoxes', 'inputs': {
            'width': ['size', 0], 'height': ['size', 1], 'editor_state': {'__value__': boxes},
            'last_incoming': {'__value__': []}, 'background': prev}}
        g[f'segs{i}'] = {'class_type': 'SAM3BoxesToSEGS', 'inputs': {
            'bboxes': [f'box{i}', 1], 'crop_factor': st.get('crop_factor', 2.0), 'padding': st.get('padding', 16), 'image': prev}}
        g[f'pos{i}'] = {'class_type': 'CLIPTextEncode', 'inputs': {'text': pos, 'clip': ['ckpt', 1]}}
        g[f'neg{i}'] = {'class_type': 'CLIPTextEncode', 'inputs': {'text': neg, 'clip': ['ckpt', 1]}}
        det_in = prev
        if i == 0 and job.get('restore_original'):
            orig_name = f'{name}-original.png'
            shutil.copy2(job['restore_original'], os.path.join(INPUT_DIR, orig_name))
            g['restore_load'] = {'class_type': 'LoadImage', 'inputs': {'image': orig_name}}
            g['restore_mask'] = {'class_type': 'SegsToCombinedMask', 'inputs': {'segs': [f'segs{i}', 0]}}
            g['restore'] = {'class_type': 'ImageCompositeMasked', 'inputs': {
                'destination': prev, 'source': ['restore_load', 0], 'x': 0, 'y': 0, 'resize_source': False, 'mask': ['restore_mask', 0]}}
            det_in = ['restore', 0]
        d = dict(DETAILER, model_name=ckpt, seed=st.get('seed', secrets.randbits(50)),
                 image=det_in, segs=[f'segs{i}', 0], vae=['ckpt', 2], positive=[f'pos{i}', 0], negative=[f'neg{i}', 0])
        if st.get('denoise') is not None:
            d['denoise'] = st['denoise']
        g[f'det{i}'] = {'class_type': 'RemoteSEGSDetailer', 'inputs': d}
        prev = [f'det{i}', 0]
    g['save'] = {'class_type': 'SaveImage', 'inputs': {'images': prev, 'filename_prefix': f'repair/{name}'}}
    return g


spec = json.load(open('/content/repair_jobs.json'))
ckpt = spec.get('ckpt', CKPT_DEFAULT)
os.makedirs(INPUT_DIR, exist_ok=True)
ids = []
for job in spec['jobs']:
    graph = build(job, ckpt)
    req = urllib.request.Request('http://127.0.0.1:8188/prompt', data=json.dumps({'prompt': graph}).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=30))
        ids.append({'name': job['name'], 'prompt_id': r['prompt_id']})
        print('queued', job['name'], r['prompt_id'][:8])
    except urllib.error.HTTPError as e:
        print('REJECTED', job['name'], e.code, e.read().decode()[:1800]); break
json.dump(ids, open('/content/repair_ids.json', 'w'))
