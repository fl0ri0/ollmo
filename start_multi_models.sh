#!/bin/bash
# Starts the manager that discovers and launches dedicated model servers
# and then starts the Flask webserver for the UI.
# Usage: run from the ollmo repo root: ./start_multi_models.sh

cd "$(dirname "$0")" # Ensure we are in the ollmo repo root.

OLLAMA_CLI="/opt/homebrew/bin/ollama"
MAIN_VENV_DIR=".venv"
REPO_PYTHON="python3"
REQUIREMENTS_FILE="requirements.txt" # In the ollmo repo root.
UNIFIED_STARTUP_SCRIPT="scripts/startup_model_manager.py"
WEBSERVER_SCRIPT="ollmo_webserver.py"
WEBSERVER_PORT="5001"

# Keep reviewed-rebase credentials out of every model/backend process. They
# are re-exported only for the Flask control plane, which immediately removes
# them from its process environment after startup.
GRAPH_REBASE_OPERATOR_TOKEN="${OLLMO_GRAPH_REBASE_OPERATOR_TOKEN:-}"
GRAPH_REBASE_OPERATOR_IDENTITY="${OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY:-}"
export -n GRAPH_REBASE_OPERATOR_TOKEN
export -n GRAPH_REBASE_OPERATOR_IDENTITY
unset OLLMO_GRAPH_REBASE_OPERATOR_TOKEN
unset OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY

: "${OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS:=4}"
: "${OLLMO_GRAPH_REPAIR_AUTONOMY=apply_enforced}"
: "${OLLMO_APPLY_ENFORCED_POLICY=safe_v1}"
export OLLMO_MULTI_MATERIALIZATION_MAX_PARALLEL_WORKERS
export OLLMO_GRAPH_REPAIR_AUTONOMY
export OLLMO_APPLY_ENFORCED_POLICY

# Colors
SYSTEM_COLOR="\033[90m"
RESET_COLOR="\033[0m"

quiet_lsof() {
    lsof -w "$@"
}

run_repo_python() {
    if [[ -x "$REPO_PYTHON" ]]; then
        "$REPO_PYTHON" "$@"
        return
    fi
    python3 "$@"
}

preflight_ollama_service_ownership() {
    if [ "${OLLMO_ALLOW_BREW_OLLAMA_SERVICE:-0}" = "1" ]; then
        echo "⚠️  OLLMO_ALLOW_BREW_OLLAMA_SERVICE=1 is set. Skipping the Homebrew Ollama ownership check."
        return 0
    fi

    if ! command -v brew >/dev/null 2>&1; then
        return 0
    fi

    local service_line
    local service_status

    service_line=$(brew services list 2>/dev/null | awk '$1=="ollama"{print; exit}')
    if [ -z "$service_line" ]; then
        return 0
    fi

    service_status=$(printf '%s\n' "$service_line" | awk '{print $2}')
    if [ "$service_status" != "started" ]; then
        return 0
    fi

    echo "❌ Homebrew service 'ollama' is already running."
    echo "   Ollmo should be the only owner of 'ollama serve'."
    echo "   Run this first: brew services stop ollama"
    echo "   If you want to override this intentionally: OLLMO_ALLOW_BREW_OLLAMA_SERVICE=1 ./start_multi_models.sh"
    return 1
}

echo -e "${SYSTEM_COLOR}--- Full startup: models + web UI ---${RESET_COLOR}"

# --- Virtual environment & dependencies ---
echo -e "\n${SYSTEM_COLOR}--- Step 1: check/activate virtual environment ---${RESET_COLOR}"
if [ ! -d "$MAIN_VENV_DIR" ]; then
    echo "🔧 Creating virtual environment ($MAIN_VENV_DIR)..."
    python3 -m venv "$MAIN_VENV_DIR"
    if [ $? -ne 0 ]; then echo "❌ Failed to create the virtual environment."; exit 1; fi
    NEEDS_INSTALL=true
else
    NEEDS_INSTALL=false
fi
# Always activate it.
if [ ! -f "$MAIN_VENV_DIR/bin/activate" ]; then echo "❌ Could not find the activate script in the virtual environment."; exit 1; fi
source "$MAIN_VENV_DIR/bin/activate"
if [[ -x "$MAIN_VENV_DIR/bin/python3" ]]; then
    REPO_PYTHON="$MAIN_VENV_DIR/bin/python3"
fi
echo "🐍 Virtual environment activated."

# Install/update dependencies.
if $NEEDS_INSTALL || { [ -f "$REQUIREMENTS_FILE" ] && [ "$REQUIREMENTS_FILE" -nt "$MAIN_VENV_DIR/bin/pip" ]; }; then
     echo "🔧 Installing/updating dependencies from $REQUIREMENTS_FILE..."
    if [ -f "$REQUIREMENTS_FILE" ]; then
        pip3 install -r "$REQUIREMENTS_FILE"
        if [ $? -ne 0 ]; then echo "❌ Failed to install dependencies."; deactivate; exit 1; fi
        touch "$MAIN_VENV_DIR/bin/pip" # Refresh the timestamp.
    else
        echo "⚠️  $REQUIREMENTS_FILE not found. Installing required packages manually..."
        pip3 install requests Flask Flask-Cors
        if [ $? -ne 0 ]; then echo "❌ Failed to install the fallback packages."; deactivate; exit 1; fi
         touch "$MAIN_VENV_DIR/bin/pip"
    fi
fi
# --- End virtual environment & dependencies ---

echo -e "\n${SYSTEM_COLOR}--- Step 1.5: check Ollama ownership ---${RESET_COLOR}"
if ! preflight_ollama_service_ownership; then
    exit 1
fi

# --- Discover/start models together ---
echo -e "\n${SYSTEM_COLOR}--- Step 2: discover and start model instances ---${RESET_COLOR}"
if [ ! -f "$UNIFIED_STARTUP_SCRIPT" ]; then
     echo "❌ Could not find $UNIFIED_STARTUP_SCRIPT."
     exit 1
fi
python3 "$UNIFIED_STARTUP_SCRIPT"
if [ $? -ne 0 ]; then
    echo "❌ Failed while running $UNIFIED_STARTUP_SCRIPT."
    exit 1
fi
echo "✅ Model startup finished."

# --- Start Flask webserver ---
echo -e "\n${SYSTEM_COLOR}--- Step 3: start the webserver for the UI ---${RESET_COLOR}"
echo "ℹ️  External provider projections are not refreshed during startup. Run './ollmo sync' manually if needed."
if [ ! -f "$WEBSERVER_SCRIPT" ]; then
     echo "❌ Could not find $WEBSERVER_SCRIPT."
     exit 1;
fi

# Check the port first.
if quiet_lsof -iTCP:$WEBSERVER_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ The webserver is already running on port $WEBSERVER_PORT."
    echo "   Dashboard:       http://127.0.0.1:$WEBSERVER_PORT/"
    echo "   Dashboard alias: http://127.0.0.1:$WEBSERVER_PORT/dashboard"
else
    echo "🚀 Starting $WEBSERVER_SCRIPT in the background on port $WEBSERVER_PORT..."
    LOG_DIR="logs"; mkdir -p "$LOG_DIR"
    FLASK_LOG="${LOG_DIR}/flask_webserver.log"
    run_repo_python - <<'PY'
from pathlib import Path

from ollmo_runtime.runtime_log_hygiene import prepare_clean_global_log

prepare_clean_global_log(
    Path("logs/flask_webserver.log"),
    metadata={
        'service': 'flask_webserver',
        'port': 5001,
    },
)
PY
    export OLLMO_GRAPH_REBASE_OPERATOR_TOKEN="$GRAPH_REBASE_OPERATOR_TOKEN"
    export OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY="$GRAPH_REBASE_OPERATOR_IDENTITY"
    nohup "$REPO_PYTHON" "$WEBSERVER_SCRIPT" > "$FLASK_LOG" 2>&1 &
    WEBSERVER_PID=$!
    unset OLLMO_GRAPH_REBASE_OPERATOR_TOKEN
    unset OLLMO_GRAPH_REBASE_OPERATOR_IDENTITY
    GRAPH_REBASE_OPERATOR_TOKEN=""
    GRAPH_REBASE_OPERATOR_IDENTITY=""
    sleep 2 # Give the server time to start.
    if quiet_lsof -iTCP:$WEBSERVER_PORT -sTCP:LISTEN -t >/dev/null ; then
        echo "✅ Webserver started successfully (PID: $WEBSERVER_PID)."
        echo "   Dashboard:       http://127.0.0.1:$WEBSERVER_PORT/"
        echo "   Dashboard alias: http://127.0.0.1:$WEBSERVER_PORT/dashboard"
        echo "   (The server is running in the background, logs in $FLASK_LOG)"
    else
        echo "❌ Failed to start the webserver. See $FLASK_LOG."
        exit 1
    fi
fi
# --- End webserver start ---

echo -e "\n${SYSTEM_COLOR}--- Startup complete ---${RESET_COLOR}"
# The virtual environment stays active for this terminal session unless you call 'deactivate'.
