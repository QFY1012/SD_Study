"""Runs ON the VM. Cancel PENDING jobs whose SaveImage prefix contains any name listed in /content/cancel_names.json
(e.g. ["rhandB", "claw190"]). A job that is already running is not interrupted."""
import json, urllib.request
names = json.load(open('/content/cancel_names.json'))
q = json.load(urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=10))
pref = lambda p: [n['inputs'].get('filename_prefix', '') for n in p[2].values() if n.get('class_type') == 'SaveImage'][0]
todel = [p[1] for p in q['queue_pending'] if any(n in pref(p) for n in names)]
if todel:
    req = urllib.request.Request('http://127.0.0.1:8188/queue', data=json.dumps({'delete': todel}).encode(), headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req, timeout=10)
print('cancelled', len(todel), 'pending jobs; kept', len(q['queue_pending']) - len(todel))
