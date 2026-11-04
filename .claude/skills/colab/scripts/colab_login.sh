#!/usr/bin/env bash
# Start the interactive `colab` login in the background and print the Google URL for the user.
# Finish with: colab_login_finish.sh <authorization-code>
set -euo pipefail
D="${COLAB_LOGIN_DIR:-/tmp/colab-login}"; mkdir -p "$D"; rm -f "$D/in" "$D/out.log"
mkfifo "$D/in"
( sleep 3000 > "$D/in" & )                               # keep the fifo open as stdin
( nohup colab --auth oauth2 usage < "$D/in" > "$D/out.log" 2>&1 & )
for _ in $(seq 1 30); do grep -q "https://accounts.google.com" "$D/out.log" 2>/dev/null && break; sleep 1; done
grep -o 'https://accounts.google.com[^ ]*' "$D/out.log" | head -1
