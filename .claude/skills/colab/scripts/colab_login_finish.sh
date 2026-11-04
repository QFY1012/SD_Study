#!/usr/bin/env bash
# usage: colab_login_finish.sh <authorization-code>
set -euo pipefail
D="${COLAB_LOGIN_DIR:-/tmp/colab-login}"
printf '%s\n' "$1" > "$D/in"; sleep 15; tail -5 "$D/out.log"
