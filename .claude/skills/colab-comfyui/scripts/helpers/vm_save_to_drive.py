"""Runs ON the VM. Copy finished outputs into NEW Drive folders (never into the user's dataset folders).
Edit the DST/SRC pairs first. Detached so the kernel is not blocked."""
import subprocess
code = r'''
import os, shutil
D = "/content/drive/MyDrive"
PAIRS = [("/content/test_out/repair", D + "/ComfyUI/output/my_repair_candidates"),
         ("/content/test_out/cheese_sports", D + "/ComfyUI/output/my_generated")]
log = open("/content/save_drive.log", "w")
for src, dst in PAIRS:
    os.makedirs(dst, exist_ok=True); n = 0
    for f in sorted(os.listdir(src)) if os.path.isdir(src) else []:
        t = os.path.join(dst, f)
        if f.endswith(".png") and not os.path.exists(t):
            shutil.copyfile(os.path.join(src, f), t); n += 1
    print(src, "->", dst, n, "files", file=log, flush=True)
print("DONE", file=log, flush=True)
'''
subprocess.Popen(['python3', '-c', code], start_new_session=True)
print('saving in background; read /content/save_drive.log')
