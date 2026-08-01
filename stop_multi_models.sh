#!/bin/bash
# Stops the Flask webserver first, then all dedicated runtime servers
# and the default Ollama server (port 11434), matching the original script's intent.

cd "$(dirname "$0")"

CONFIG_FILE="model_ports.json"
START_PORT=11435
END_PORT=11500
DEFAULT_PORT=11434
MLX_START_PORT=${MLX_START_PORT:-11501}
MLX_PORT_MAX=${MLX_PORT_MAX:-11550}
LLAMA_CPP_START_PORT=${LLAMA_CPP_START_PORT:-11551}
LLAMA_CPP_PORT_MAX=${LLAMA_CPP_PORT_MAX:-11600}
WEBSERVER_SCRIPT="ollmo_webserver.py" # Webserver script name.

quiet_lsof() {
    lsof -w "$@"
}

run_repo_python() {
    if [[ -x "./.venv/bin/python3" ]]; then
        "./.venv/bin/python3" "$@"
        return
    fi
    python3 "$@"
}

print_llama_server_processes() {
    local output
    output=$(ps aux | awk '
        /[l]lama-server/ {
            line = $1
            for (i = 2; i <= 10 && i <= NF; i++) {
                line = line " " $i
            }
            for (i = 11; i <= NF; i++) {
                line = line " " $i
                if ($i == "--port" && (i + 1) <= NF) {
                    line = line " " $(i + 1)
                    break
                }
            }
            print line
        }
    ')
    if [ -n "$output" ]; then
        printf '%s\n' "$output"
    else
        echo "No 'llama-server' processes found."
    fi
}

add_pid_if_new() {
    local candidate="$1"
    for existing_pid in "${PIDS_TO_KILL[@]}"; do
        if [[ "$existing_pid" == "$candidate" ]]; then
            return
        fi
    done
    PIDS_TO_KILL+=("$candidate")
}

REGISTRY_RUNTIME_PORTS=()

add_runtime_port_if_new() {
    local candidate="$1"
    if ! [[ "$candidate" =~ ^[0-9]+$ ]]; then
        return
    fi
    for existing_port in "${REGISTRY_RUNTIME_PORTS[@]}"; do
        if [[ "$existing_port" == "$candidate" ]]; then
            return
        fi
    done
    REGISTRY_RUNTIME_PORTS+=("$candidate")
}

collect_registry_runtime_ports() {
    if [ ! -f "$CONFIG_FILE" ]; then
        return
    fi

    if command -v jq &> /dev/null; then
        while IFS= read -r port; do
            add_runtime_port_if_new "$port"
        done < <(jq -r '.[]? | select(type=="object" and (.agent != true)) | .port // empty' "$CONFIG_FILE" 2>/dev/null)
        return
    fi

    while IFS= read -r port; do
        add_runtime_port_if_new "$port"
    done < <(python3 - "$CONFIG_FILE" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    data = []

entries = data if isinstance(data, list) else data.get("models") or data.get("instances") or []
for entry in entries:
    if not isinstance(entry, dict):
        continue
    if entry.get("agent"):
        continue
    port = entry.get("port")
    try:
        print(int(port))
    except (TypeError, ValueError):
        continue
PY
)
}

build_confirmed_stopped_ports_csv() {
    local stopped_ports=()
    for port in "${REGISTRY_RUNTIME_PORTS[@]}"; do
        if ! quiet_lsof -iTCP:${port} -sTCP:LISTEN -t >/dev/null 2>&1; then
            stopped_ports+=("$port")
        fi
    done
    local csv=""
    local first=1
    for port in "${stopped_ports[@]}"; do
        if [ "$first" -eq 1 ]; then
            csv="$port"
            first=0
        else
            csv="${csv},${port}"
        fi
    done
    printf '%s' "$csv"
}

scan_port_range() {
    local range_start="$1"
    local range_end="$2"
    local label="$3"
    local found_flag=0

    if [[ -z "$range_start" || -z "$range_end" ]]; then
        return
    fi

    if ! [[ "$range_start" =~ ^[0-9]+$ && "$range_end" =~ ^[0-9]+$ ]]; then
        echo -e "\n2. Skipping range ${range_start}-${range_end} (${label}) because the values are invalid."
        return
    fi

    if [ "$range_start" -gt "$range_end" ]; then
        local tmp="$range_start"
        range_start="$range_end"
        range_end="$tmp"
    fi

    echo -e "\n2. Searching for servers on ports ${range_start}-${range_end} (${label})..."
    for port in $(seq "$range_start" "$range_end"); do
        local pid_on_port
        pid_on_port=$(quiet_lsof -iTCP:${port} -sTCP:LISTEN -t 2>/dev/null)
        if [ -n "$pid_on_port" ]; then
            found_flag=1
            echo "   Found a server on port ${port} (PID: ${pid_on_port}). Adding it to the stop list."
            add_pid_if_new "$pid_on_port"
        fi
    done

    if [[ "$found_flag" -eq 0 ]]; then
        echo "   No running servers found in that range."
    fi
}

# --- Step 0: stop webserver ---
echo "--- Stop webserver & model instances ---"

# --- Step 1: stop the Flask webserver ---
echo -e "\n1. Stopping webserver ($WEBSERVER_SCRIPT)..."
WEBSERVER_PIDS=$(pgrep -f "$WEBSERVER_SCRIPT")
if [ -n "$WEBSERVER_PIDS" ]; then
    echo "   Stopping Flask webserver (PIDs: $WEBSERVER_PIDS)..."
    kill $WEBSERVER_PIDS # Gentle attempt first.
    sleep 1 # Give it a moment.
    for pid in $WEBSERVER_PIDS; do # Hard kill if it is still alive.
        if ps -p $pid > /dev/null; then echo "      PID $pid is still running, sending kill -9..."; kill -9 "$pid"; fi
    done
    echo "   Webserver stopped."
else
    echo "   No running Flask webserver ($WEBSERVER_SCRIPT) found."
fi
# --- End webserver stop ---


# === Original script flow continues here from the status check ===
echo -e "\n\n===== Current runtime status (before stopping) ====="
echo "-> Running 'ollama serve' processes (ps aux):"
ps aux | grep '[o]llama serve' || echo "No 'ollama serve' processes found."
echo -e "\n-> Running 'llama-server' processes (ps aux):"
print_llama_server_processes
echo -e "\n-> Runtime-related ports before stopping:"
quiet_lsof -i -P | grep ollama || echo "No open Ollama ports found."
quiet_lsof -i -P | grep mlx || echo "No open MLX ports found."
echo -e "\n-> Ollama instance ports (${START_PORT}-${END_PORT}):"
quiet_lsof -iTCP:${START_PORT}-${END_PORT} -sTCP:LISTEN -P || echo "No servers found in the Ollama instance port range."
echo -e "\n-> llama.cpp instance ports (${LLAMA_CPP_START_PORT}-${LLAMA_CPP_PORT_MAX}):"
quiet_lsof -iTCP:${LLAMA_CPP_START_PORT}-${LLAMA_CPP_PORT_MAX} -sTCP:LISTEN -P || echo "No llama.cpp servers found in the llama.cpp port range."
echo "================================================"

collect_registry_runtime_ports
if [ ${#REGISTRY_RUNTIME_PORTS[@]} -gt 0 ]; then
    echo "-> Snapshot runtime ports from model_ports.json: ${REGISTRY_RUNTIME_PORTS[*]}"
else
    echo "-> Snapshot runtime ports from model_ports.json: none"
fi

PIDS_TO_KILL=()

# --- Step 1: try to read PIDs from the JSON file ---
if [ -f "$CONFIG_FILE" ]; then
    echo -e "\n1. Trying to read PIDs from '$CONFIG_FILE'..."
    if command -v jq &> /dev/null; then
        PIDS_FROM_JSON=$(jq -r '.[].pid // empty | select(. != null)' "$CONFIG_FILE" 2>/dev/null)
        if [ -n "$PIDS_FROM_JSON" ]; then
            echo "   Found PIDs from JSON: $PIDS_FROM_JSON"
            for pid in $PIDS_FROM_JSON; do
                 if [[ "$pid" =~ ^[0-9]+$ ]]; then PIDS_TO_KILL+=($pid); fi
            done
        else
            echo "   No valid PIDs found in '$CONFIG_FILE', or the file is empty."
        fi
    else
        echo "   ⚠️  'jq' was not found. PIDs cannot be read from '$CONFIG_FILE'."
    fi
else
    echo -e "\n1. Configuration file '$CONFIG_FILE' not found. Skipping PID extraction."
fi

# --- Step 2: search dedicated ports (fallback/cleanup) ---
scan_port_range "$START_PORT" "$END_PORT" "dedicated Ollama range"
scan_port_range "$MLX_START_PORT" "$MLX_PORT_MAX" "MLX range"
scan_port_range "$LLAMA_CPP_START_PORT" "$LLAMA_CPP_PORT_MAX" "llama.cpp range"

# --- Step 3: handle the default server ---
echo -e "\n3. Checking default server (port ${DEFAULT_PORT})..."
DEFAULT_PID=$(quiet_lsof -iTCP:${DEFAULT_PORT} -sTCP:LISTEN -t 2>/dev/null)
if [ -n "$DEFAULT_PID" ]; then
    echo "   Found the default server on port ${DEFAULT_PORT} (PID: ${DEFAULT_PID}). Adding it to the stop list."
    found=0
     for existing_pid in "${PIDS_TO_KILL[@]}"; do if [[ "$existing_pid" == "$DEFAULT_PID" ]]; then found=1; break; fi; done
     if [[ "$found" -eq 0 ]]; then PIDS_TO_KILL+=($DEFAULT_PID); fi
else
    echo "   The default server on port ${DEFAULT_PORT} is not running."
fi

# --- Step 4: stop runtime processes ---
UNIQUE_PIDS=($(echo "${PIDS_TO_KILL[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
if [ ${#UNIQUE_PIDS[@]} -gt 0 ]; then
    echo -e "\n4. Stopping these runtime server PIDs: ${UNIQUE_PIDS[@]}"
    killed_count=0
    for pid in "${UNIQUE_PIDS[@]}"; do
         if ps -p $pid > /dev/null; then
              echo "   Stopping PID $pid..."
              kill "$pid"; sleep 0.5
              if ps -p $pid > /dev/null; then echo "      PID $pid is still running, using kill -9..."; kill -9 "$pid"; fi
              killed_count=$((killed_count + 1))
         else
              echo "   PID $pid no longer exists."
         fi
    done
     echo "   Sent stop signals to $killed_count runtime server process(es)."
else
    echo -e "\n4. No running runtime server processes found to stop."
fi

# --- Step 5: final status check ---
echo -e "\n\n===== Current status (after stopping) ====="
echo "-> Running 'ollama serve' processes (ps aux):"
ps aux | grep '[o]llama serve' || echo "No 'ollama serve' processes remain."
echo -e "\n-> Running 'llama-server' processes (ps aux):"
print_llama_server_processes | sed "s/No 'llama-server' processes found./No 'llama-server' processes remain./"
echo -e "\n-> Webserver status:"
WEBSERVER_PID=$(quiet_lsof -iTCP -sTCP:LISTEN -nP | grep "$WEBSERVER_SCRIPT" | awk '{print $2}' | head -n 1)
if [ -n "$WEBSERVER_PID" ]; then
    echo "   ⚠️  The webserver is still running (PID: $WEBSERVER_PID)"
else
    echo "No running webserver process remains."
fi
echo -e "\n-> Runtime-related ports:"
quiet_lsof -i -P | grep ollama || echo "No open Ollama ports remain."
quiet_lsof -i -P | grep mlx || echo "No open MLX ports remain."
quiet_lsof -iTCP:${LLAMA_CPP_START_PORT}-${LLAMA_CPP_PORT_MAX} -sTCP:LISTEN -P || echo "No open llama.cpp ports remain."

echo "================================================"

# Finalize runtime-state files via the shared hygiene helper.
echo -e "\n🧹 Finalizing runtime registry, status, and log hygiene..."
run_repo_python - <<'PY'
from pathlib import Path

from ollmo_core.status import DEFAULT_RUNTIME_STATUS_PATH
from ollmo_runtime.runtime_hygiene import finalize_runtime_shutdown

summary = finalize_runtime_shutdown(
    registry_path=Path("model_ports.json"),
    status_path=DEFAULT_RUNTIME_STATUS_PATH,
    log_dir=Path("logs"),
    sync_external=False,
    preserve_agents=True,
)
print(
    "   Runtime shutdown complete: "
    f"{summary.get('runtime_status_count', 0)} runtime-status entries still active, "
    f"{summary.get('archived_count', 0)} log file(s) archived."
)
PY

echo -e "\n--- Stop script finished ---"
echo "ℹ️  External provider projections were left untouched. Run './ollmo sync' or the cleanup/unsync scripts manually if needed."

exit 0
