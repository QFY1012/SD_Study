#!/usr/bin/env bash
# Send the Enter the mount is waiting for (after the user approved) and show the result.
set -euo pipefail
D="${COLAB_MOUNT_DIR:-/tmp/colab-mount}"
echo "" > "$D/in"; sleep 20; tail -3 "$D/out.log"
