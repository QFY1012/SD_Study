---
name: colab
description: Start and use a Google Colab GPU runtime from the terminal with the `colab` CLI (google-colab-cli) - install, log in, create a session, mount Drive, run code, upload/download files, stop. Use when the user wants something run on their Colab from here, or a Colab runtime started/stopped.
---

# colab - start and use Colab from the terminal

Requirements: Linux or macOS (Windows needs WSL), `uv`, network access to Google. The CLI talks to Colab directly; no browser tab is needed on the machine that runs it.

## Rules
- Only use sessions **you created** (give them a name, e.g. `work`). `colab sessions` may also list the user's own runtimes - never exec/stop those.
- Ask before starting a GPU (it spends the user's compute units, about 1.15 units/h for a T4) and stop the session when the work is finished: `colab stop -s <name>`.
- Login grants broad Google permissions (`cloud-platform`, `colaboratory`, `drive.file`). Say so, and when done: delete `~/.config/colab-cli/token.json` and tell the user to revoke the grant at https://myaccount.google.com/permissions.

## 1. Install
```bash
uv tool install google-colab-cli      # needs Python 3.13; uv fetches it
colab version
```

## 2. Log in (once; needs the user for ~1 minute)
`scripts/colab_login.sh` starts the login and prints a Google URL. Give the URL to the user; they approve in a browser and send back the authorization code. Then `scripts/colab_login_finish.sh <code>`. Check with `colab usage` (prints compute-unit balance).

## 3. Create / list / stop a session
```bash
colab new -s work --gpu T4        # T4, L4, G4, A100, H100; omit --gpu for CPU; --high-mem needs Pro
colab sessions
colab status -s work
colab restart-kernel -s work
colab stop -s work
```
`colab url -s work` prints a browser link, but it did **not** attach to the running VM when the user opened it (blank notebook, no GPU). Do not rely on it.

## 4. Mount Drive (needs a second consent per runtime)
`scripts/colab_drivemount.sh work` starts the mount and prints another Google URL; the user approves, then run `scripts/colab_drivemount_finish.sh` (sends the Enter it waits for). Running `colab drivemount` with stdin closed fails with HTTP 400. The Drive appears at `/content/drive/MyDrive`.

## 5. Run code, move files
```bash
colab exec -s work -f script.py           # run a local script on the VM
echo 'print(1)' | colab repl -s work      # one-liners
colab install -s work -r requirements.txt # pip on the VM
colab upload   -s work LOCAL /content/REMOTE
colab download -s work /content/REMOTE LOCAL
colab ls -s work /content
colab log -s work -o history.ipynb        # replayable log
```

## 6. Things that bite
- `colab exec` gives up after about a minute (`TimeoutError: Timeout waiting for reply`) but the VM code **keeps running**. Never put long work inside an exec: start it detached (`scripts/vm_detach.py`) and poll (`scripts/vm_poll.py`).
- The VM kernel is single-threaded: a long exec **blocks every later exec** until it finishes. Heavy Drive reads (thousands of MB) should also be run detached.
- Reading big files from the Drive mount is slow the first time (a 6.5 GB file took ~3.5 minutes); copy once to `/content` if you will read it repeatedly.
- Shell tools that run commands for you may auto-background anything over ~2 minutes; use a background run plus a polling loop and wait for its notification.
- `colab upload` of a ~75 MB file takes under a minute; much larger files are better put on Drive.
- Never type or pipe the user's codes/tokens anywhere except the login prompt they were issued for.

## 7. Cleanup checklist
`colab stop -s <your session>` -> `colab sessions` shows none of yours -> delete `~/.config/colab-cli/token.json` -> remind the user to revoke the Google grant.
