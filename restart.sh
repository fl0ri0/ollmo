#!/bin/bash
# Convenience script to fully restart the local OLLMO stack.
# 1. Runs stop_multi_models.sh and waits for completion
# 2. Runs start_multi_models.sh
# 3. Waits for the UI port to become available
# 4. Opens http://127.0.0.1:5001 in the default browser (macOS `open`)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
STOP_SCRIPT="$ROOT_DIR/stop_multi_models.sh"
START_SCRIPT="$ROOT_DIR/start_multi_models.sh"
UI_PORT=5001
UI_URL="http://127.0.0.1:${UI_PORT}"

wait_for_port() {
    local tries=40
    local delay=1
    for ((i=1; i<=tries; i++)); do
        if python3 - "$UI_PORT" >/dev/null 2>&1 <<'PY'
import socket, sys
port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    sys.exit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
        then
            return 0
        fi
        sleep "$delay"
    done
    return 1
}

echo "🔻 Stopping existing stack via $STOP_SCRIPT..."
if ! bash "$STOP_SCRIPT"; then
    echo "❌ stop_multi_models.sh failed. Aborting restart."
    exit 1
fi

echo "🚀 Starting stack via $START_SCRIPT..."
if ! bash "$START_SCRIPT"; then
    echo "❌ start_multi_models.sh failed. Aborting restart."
    exit 1
fi

echo "⏳ Waiting for UI on port $UI_PORT..."
if wait_for_port; then
    echo "✅ UI reachable at $UI_URL"
else
    echo "⚠️  UI did not become reachable; opening browser anyway."
fi

if command -v open >/dev/null 2>&1; then
    echo "🌐 Opening $UI_URL in default browser..."
    open "$UI_URL"
else
    echo "ℹ️  Please open $UI_URL in your browser."
fi
