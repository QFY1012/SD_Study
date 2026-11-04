"""Runs ON the Colab VM. Prints install / server / job status in one go (always returns quickly)."""
import json, os, urllib.request

print('install:', 'done' if os.path.exists('/content/install.done') else 'FAILED' if os.path.exists('/content/install.fail') else 'running/not started')
print('drive mounted:', os.path.isdir('/content/drive/MyDrive'))
try:
    s = json.load(urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3))
    print('comfyui: up,', [d['name'] for d in s['devices']])
    q = json.load(urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=3))
    print('queue: running', len(q['queue_running']), 'pending', len(q['queue_pending']))
    if os.path.exists('/content/last_prompt_id'):
        pid = open('/content/last_prompt_id').read()
        h = json.load(urllib.request.urlopen('http://127.0.0.1:8188/history/' + pid, timeout=5))
        if pid in h:
            e = h[pid]
            print('last job:', e['status']['status_str'])
            for m in e['status']['messages']:
                if m[0] == 'execution_error':
                    print('ERROR', json.dumps(m[1])[:1500])
        else:
            print('last job: not finished yet')
except Exception as ex:
    print('comfyui: not reachable (' + type(ex).__name__ + ')')
if os.path.exists('/content/comfy.log'):
    print('--- comfy.log tail ---')
    print(open('/content/comfy.log', errors='replace').read()[-800:])
