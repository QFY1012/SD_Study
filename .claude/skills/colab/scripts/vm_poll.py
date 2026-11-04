"""Runs ON the VM. Prints whether the detached job finished (and its exit code) plus the last lines of its log."""
import os
done = os.path.exists('/content/job.done')
print('finished, exit code', open('/content/job.done').read().strip() if done else 'still running')
if os.path.exists('/content/job.log'):
    print('--- job.log (tail) ---'); print(open('/content/job.log', errors='replace').read()[-1500:])
