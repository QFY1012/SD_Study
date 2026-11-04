"""Runs ON the VM. One JSON line: ComfyUI progress (print it in a local `for` loop with sleep to watch a batch)."""
import json, re, urllib.request
d = json.load(urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5))
pref = lambda p: [n['inputs'].get('filename_prefix', '') for n in p[2].values() if n.get('class_type') == 'SaveImage'][0].split('/')[-1]
log = re.sub(r'\x1b\[[0-9;]*m', '', open('/content/comfy.log', errors='replace').read()[-500:])
last = [l for l in log.splitlines() if 'it/s' in l or 's/it' in l]
print(json.dumps({'running': [pref(p) for p in d['queue_running']], 'pending': len(d['queue_pending']), 'step': last[-1].strip()[:70] if last else ''}))
