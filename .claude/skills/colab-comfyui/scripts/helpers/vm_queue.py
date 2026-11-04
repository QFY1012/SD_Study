"""Runs ON the VM. Show what ComfyUI is running / has queued, and how many repair outputs exist per job name."""
import collections, json, os, urllib.request
q = json.load(urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=10))
pref = lambda p: [n['inputs'].get('filename_prefix', '') for n in p[2].values() if n.get('class_type') == 'SaveImage'][0]
print('running:', [pref(p) for p in q['queue_running']])
print('pending:', dict(collections.Counter(pref(p).split('_s')[0] for p in q['queue_pending'])))
d = '/content/test_out/repair'
if os.path.isdir(d):
    print('outputs:', dict(collections.Counter(f.split('_s')[0] for f in os.listdir(d) if f.endswith('.png'))))
