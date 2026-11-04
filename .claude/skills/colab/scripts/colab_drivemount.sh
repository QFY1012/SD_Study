#!/usr/bin/env bash
# usage: colab_drivemount.sh <session>   -> prints the Google URL the user must approve.
# Then run colab_drivemount_finish.sh.
set -euo pipefail
S="${1:?session name}"; D="${COLAB_MOUNT_DIR:-/tmp/colab-mount}"; mkdir -p "$D"; rm -f "$D/in" "$D/out.log"
mkfifo "$D/in"; ( sleep 3000 > "$D/in" & )
( nohup colab drivemount -s "$S" < "$D/in" > "$D/out.log" 2>&1 & )
for _ in $(seq 1 40); do grep -q "https://accounts.google.com" "$D/out.log" 2>/dev/null && break; sleep 1; done
grep -o 'https://accounts.google.com[^ ]*' "$D/out.log" | head -1
