"""Runs ON the Colab VM (via `colab exec -f`). Clones ComfyUI and starts `pip install` in the
background (exec calls time out after ~1 minute, so long work must be detached). Poll with vm_status.py."""
import os, subprocess, sys
from pathlib import Path

C = Path('/content/ComfyUI')
if not C.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/Comfy-Org/ComfyUI.git', str(C)], check=True)
for d in ('/content/test_input', '/content/test_out', str(C / 'custom_nodes/ComfyUI-SEGSInpaint')):
    os.makedirs(d, exist_ok=True)
for f in ('/content/install.done', '/content/install.fail'):
    Path(f).unlink(missing_ok=True)
cmd = (f'{sys.executable} -m pip install -q -r {C}/requirements.txt > /content/install.log 2>&1 '
       f'&& touch /content/install.done || touch /content/install.fail')
subprocess.Popen(['bash', '-c', cmd], start_new_session=True)
print('install started in background; poll with vm_status.py')
