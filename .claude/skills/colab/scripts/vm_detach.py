"""Runs ON the VM (colab exec -f vm_detach.py). Starts the shell command in /content/cmd.txt detached,
logging to /content/job.log and creating /content/job.done (exit code inside) when it ends."""
import subprocess
cmd = open('/content/cmd.txt').read().strip()
wrapped = f'( {cmd} ) > /content/job.log 2>&1; echo $? > /content/job.done'
subprocess.run('rm -f /content/job.done', shell=True)
subprocess.Popen(['bash', '-c', wrapped], start_new_session=True)
print('started:', cmd)
