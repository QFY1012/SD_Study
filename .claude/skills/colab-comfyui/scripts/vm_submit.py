"""Runs ON the Colab VM. Submits /content/test_wf.json (API format) and records the prompt id."""
import json, urllib.error, urllib.request

import os
wf = json.load(open(os.environ.get("WF", "/content/test_wf.json")))
req = urllib.request.Request('http://127.0.0.1:8188/prompt', data=json.dumps({'prompt': wf}).encode(),
                             headers={'Content-Type': 'application/json'})
try:
    r = json.load(urllib.request.urlopen(req, timeout=30))
    open('/content/last_prompt_id', 'w').write(r['prompt_id'])
    print('submitted', r['prompt_id'], 'node_errors:', r.get('node_errors'))
except urllib.error.HTTPError as e:
    print('REJECTED', e.code, e.read().decode()[:2500])
