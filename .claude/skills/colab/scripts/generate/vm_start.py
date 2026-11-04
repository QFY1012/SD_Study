"""Runs ON the Colab VM. Starts ComfyUI (models from Drive) detached and waits for it. Idempotent."""
import json, os, subprocess, sys, time, urllib.request

DRIVE_MODELS = os.environ.get('DRIVE_MODELS', '/content/drive/MyDrive/ComfyUI/models')
INPUT_DIR = os.environ.get('COMFY_INPUT', '/content/test_input')
OUTPUT_DIR = os.environ.get('COMFY_OUTPUT', '/content/test_out')
assert os.path.isdir(DRIVE_MODELS), 'Drive not mounted or models dir missing: ' + DRIVE_MODELS


def up():
    try:
        return json.load(urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3))
    except Exception:
        return None


if up():
    print('already running')
else:
    open('/content/extra_model_paths.yaml', 'w').write(f'''drive:
  base_path: {DRIVE_MODELS}
  checkpoints: checkpoints
  loras: loras
  vae: vae
  clip: clip
  text_encoders: text_encoders
  diffusion_models: diffusion_models
''')
    p = subprocess.Popen([sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8188',
                          '--extra-model-paths-config', '/content/extra_model_paths.yaml',
                          '--input-directory', INPUT_DIR, '--output-directory', OUTPUT_DIR],
                         cwd='/content/ComfyUI', stdout=open('/content/comfy.log', 'w'),
                         stderr=subprocess.STDOUT, start_new_session=True)
    for i in range(25):  # stay under the exec timeout; call again if still starting
        if p.poll() is not None:
            print('EXITED', p.returncode); print(open('/content/comfy.log').read()[-3000:]); sys.exit(1)
        if up():
            break
        time.sleep(2)
s = up()
print('comfyui:', 'UP ' + str([d['name'] for d in s['devices']]) if s else 'still starting; run vm_status.py')
