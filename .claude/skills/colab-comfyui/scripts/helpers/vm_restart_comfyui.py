import subprocess, time
subprocess.run(['pkill','-f','main.py --listen'])
time.sleep(4)
print(subprocess.run("pgrep -fc 'main.py --listen' || true",shell=True,capture_output=True,text=True).stdout.strip(),'processes left')
