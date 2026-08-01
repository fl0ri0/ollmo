#!/bin/bash
# Repo-local reset helper for Ollmo.
# `clean` removes generated/runtime ballast.
# `archiv` / `archive` first archives the same useful ballast into .ollmo_archiv/<timestamp>/
# and then cleans the live runtime paths. The artifacts tree is copied before live cleanup so
# standard artifact bucket directories can stay in place.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_BASE_REL=".ollmo_archiv"

RESET_REGISTRY=0
FORGET_GHOST=0
RESET_LLAMA_CATALOG=0
DRY_RUN=0
ARCHIVE_MODE=0
FULL_RESET=0
RESET_CHROME_FILE_ACCESS_PROMPT=0

ARCHIVE_RUN_REL=""
ARCHIVE_RUN_DIR=""
LEARNING_RETENTION_STATUS="not_run"
LEARNING_RETAINED_SIDECAR_COUNT=0
LEARNING_MISSING_SIDECAR_COUNT=0
LEARNING_UNSAFE_REF_COUNT=0
LEARNING_RETENTION_FAILED=0
READINESS_RETENTION_STATUS="not_run"
READINESS_SELECTED_OBSERVATION_COUNT=0
READINESS_SETTLED_OBSERVATION_COUNT=0
READINESS_REGISTERED_OBSERVATION_COUNT=0
READINESS_MISSING_SETTLED_OBSERVATION_COUNT=0
READINESS_ACTIVE_OBSERVATION_COUNT=0
READINESS_SCAN_ERROR_COUNT=0
READINESS_HYDRATION_ERROR_COUNT=0
READINESS_REGISTRY_ERROR_COUNT=0
READINESS_ERROR_COUNT=0
READINESS_APPENDED_RECORD_COUNT=0
READINESS_RETENTION_FAILED=0

usage() {
    cat <<'USAGE'
Usage: clean_repo_state.sh [options]

Options:
  --archiv, --archive   Archive useful cleanup ballast into .ollmo_archiv/<timestamp>/ before cleaning.
                         The artifacts tree is copied first; live cleanup then removes contents
                         while preserving standard artifact bucket dirs; missing ones are
                         created only as fallback.
  --full, --empty-state  Full reset shortcut. In clean mode, also forget Ghost preferences/memory.
                         In archive mode, reset registry/catalog but preserve protected Ghost state.
  --dry-run             Show what would happen without deleting or moving anything.
  --reset-registry      Reset repo-local model_ports.json to [] after optional archiving.
  --forget-ghost        Remove Ghost preferences and compiled-memory residue after optional archiving.
  --reset-llama-catalog Remove state/llama_cpp_catalog.json after optional archiving.
  --reset-chrome-file-access-prompt
                        macOS only: reset Chrome's Desktop Folder permission so the next
                        local file:// bundle preview prompts for user approval again.
  -h, --help            Show this help.

Default cleanup removes repo-local generated/runtime ballast:
  - contents under artifacts/* buckets while preserving the standard empty artifact bucket
    directories, including artifacts/bundles/. Missing standard directories are created
    only as a fallback.
  - logs/*
  - state/chat_history/*
  - state/events.jsonl
  - state/infer_history.jsonl
  - state/artifact_registry.jsonl
  - state/generated_image_provenance.jsonl
  - state/response_frames/* (compact ledger, current index, and sidecar snapshots)
    after learning-retained sidecars reachable from state/self_learning/ are copied
    into state/self_learning/retained_sidecars/ and recorded in retention_manifest.json,
    and after every selected graph-rebase readiness observation is settled and retained
    in state/graph_rebase/readiness_observations.jsonl
  - state/runtime_status.json
  - .pytest_cache, __pycache__, *.pyc, *.pyo, .DS_Store, .coverage, htmlcov, .mypy_cache, .ruff_cache

Default cleanup preserves:
  - model_ports.json
  - state/llama_cpp_catalog.json
  - state/ghost_preferences.json
  - state/ghost_compiled_memory.json
  - state/ghost_compiled_memory.md
  - state/self_learning/
  - state/self_learning/retained_sidecars/ generated from active learning refs
  - state/graph_rebase/readiness_observations.jsonl

In clean mode, `--full` / `--empty-state` is equivalent to:
  - --forget-ghost --reset-registry --reset-llama-catalog

In archive mode, `--full` / `--empty-state` snapshots protected Ghost state when present,
but does not remove active protected Ghost files. Use `--forget-ghost` explicitly to remove
Ghost preferences/compiled-memory residue after archiving. Use
`python3 scripts/ollmoctl.py ghost --reset-learning-state --json` for self-learning reset.

`--archiv` / `--archive` keeps the same live cleanup end state, but stores archived data under:
  - .ollmo_archiv/<timestamp>/

macOS/Chrome note:
  After cleanup/archive preserves the standard artifacts/ bucket structure, Chrome may
  require Desktop Folder or Full Disk Access permission to be re-granted before local
  file:// bundle previews can load assets/css and assets/images. If Chrome shows
  net::ERR_ACCESS_DENIED for bundle assets,
  quit Chrome and re-grant the permission, or reset it with:
    tccutil reset SystemPolicyDesktopFolder com.google.Chrome
  To make that prompt happen on the next Chrome file access, run clean/archive with:
    --reset-chrome-file-access-prompt
USAGE
}

log() {
    printf '%s\n' "$*"
}

timestamp_utc() {
    date -u +"%Y%m%dT%H%M%SZ"
}

dir_has_entries() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        return 1
    fi
    find "$dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .
}

remove_dir_contents() {
    local rel="$1"
    local target="$ROOT_DIR/$rel"
    if [[ ! -d "$target" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] clear directory contents: $rel"
        return 0
    fi
    find "$target" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
}

remove_path() {
    local rel="$1"
    local target="$ROOT_DIR/$rel"
    if [[ ! -e "$target" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] remove: $rel"
        return 0
    fi
    rm -rf "$target"
}

ensure_dir() {
    local rel="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] ensure dir: $rel"
        return 0
    fi
    mkdir -p "$ROOT_DIR/$rel"
}

write_json_array() {
    local rel="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] write [] to: $rel"
        return 0
    fi
    printf '[]\n' > "$ROOT_DIR/$rel"
}

is_standard_artifact_top_level() {
    case "$1" in
        audio|audits|images|ocr|transcripts|documents|manifests|bundles|benchmarks|settings|inputs)
            return 0
            ;;
    esac
    return 1
}

is_standard_artifact_input_child() {
    case "$1" in
        audio|image|pdf|text)
            return 0
            ;;
    esac
    return 1
}

artifact_content_dirs() {
    cat <<'DIRS'
artifacts/audio
artifacts/audits
artifacts/images
artifacts/ocr
artifacts/transcripts
artifacts/documents
artifacts/manifests
artifacts/bundles
artifacts/benchmarks
artifacts/settings
artifacts/inputs/audio
artifacts/inputs/image
artifacts/inputs/pdf
artifacts/inputs/text
DIRS
}

ensure_artifact_base_dirs() {
    ensure_dir "artifacts/audio"
    ensure_dir "artifacts/audits"
    ensure_dir "artifacts/images"
    ensure_dir "artifacts/ocr"
    ensure_dir "artifacts/transcripts"
    ensure_dir "artifacts/documents"
    ensure_dir "artifacts/manifests"
    ensure_dir "artifacts/bundles"
    ensure_dir "artifacts/benchmarks"
    ensure_dir "artifacts/settings"
    ensure_dir "artifacts/inputs/audio"
    ensure_dir "artifacts/inputs/image"
    ensure_dir "artifacts/inputs/pdf"
    ensure_dir "artifacts/inputs/text"
}

stop_listener_on_port() {
    local port="$1"
    local pids
    pids="$(lsof -nP -iTCP:${port} -sTCP:LISTEN -t 2>/dev/null | sort -u || true)"
    if [[ -z "$pids" ]]; then
        return 0
    fi

    log "Stopping listener(s) on port $port: $pids"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi

    kill $pids 2>/dev/null || true
    sleep 1
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

remove_cache_dirs() {
    local pattern="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] remove directories named $pattern recursively"
        return 0
    fi
    find "$ROOT_DIR" -type d -name "$pattern" -prune -exec rm -rf {} +
}

remove_cache_files() {
    local pattern="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] remove files named $pattern recursively"
        return 0
    fi
    find "$ROOT_DIR" -type f -name "$pattern" -delete
}

archive_dir_contents() {
    local rel="$1"
    local source="$ROOT_DIR/$rel"
    local dest="$ARCHIVE_RUN_DIR/$rel"

    if ! dir_has_entries "$source"; then
        return 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] archive directory contents: $rel -> $ARCHIVE_RUN_REL/$rel/"
        return 0
    fi
    mkdir -p "$dest"
    find "$source" -mindepth 1 -maxdepth 1 -exec mv {} "$dest"/ \;
}

archive_artifacts_preserving_base_dirs() {
    local artifacts_root="$ROOT_DIR/artifacts"
    if [[ ! -d "$artifacts_root" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] copy artifacts tree before cleanup: artifacts -> $ARCHIVE_RUN_REL/artifacts/"
        return 0
    fi
    mkdir -p "$ARCHIVE_RUN_DIR"
    cp -R -p "$artifacts_root" "$ARCHIVE_RUN_DIR/"
}

remove_artifacts_preserving_base_dirs() {
    local artifacts_root="$ROOT_DIR/artifacts"
    if [[ ! -d "$artifacts_root" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] clear artifacts contents while preserving standard bucket dirs"
    fi

    while IFS= read -r rel; do
        remove_dir_contents "$rel"
    done < <(artifact_content_dirs)

    if [[ -d "$artifacts_root/inputs" ]]; then
        find "$artifacts_root/inputs" -mindepth 1 -maxdepth 1 -print 2>/dev/null | while IFS= read -r entry; do
            local name
            name="$(basename "$entry")"
            if is_standard_artifact_input_child "$name"; then
                continue
            fi
            if [[ "$DRY_RUN" -eq 1 ]]; then
                log "[dry-run] remove non-standard artifact input entry: artifacts/inputs/$name"
            else
                rm -rf "$entry"
            fi
        done
    fi

    find "$artifacts_root" -mindepth 1 -maxdepth 1 -print 2>/dev/null | while IFS= read -r entry; do
        local name
        name="$(basename "$entry")"
        if is_standard_artifact_top_level "$name"; then
            continue
        fi
        if [[ "$DRY_RUN" -eq 1 ]]; then
            log "[dry-run] remove non-standard artifact entry: artifacts/$name"
        else
            rm -rf "$entry"
        fi
    done
}

archive_path() {
    local rel="$1"
    local source="$ROOT_DIR/$rel"
    local dest="$ARCHIVE_RUN_DIR/$rel"

    if [[ ! -e "$source" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] archive path: $rel -> $ARCHIVE_RUN_REL/$rel"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    mv "$source" "$dest"
}

snapshot_path() {
    local rel="$1"
    local source="$ROOT_DIR/$rel"
    local dest="$ARCHIVE_RUN_DIR/$rel"

    if [[ ! -e "$source" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] snapshot protected path: $rel -> $ARCHIVE_RUN_REL/$rel"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    cp -R -p "$source" "$dest"
}

write_archive_manifest() {
    if [[ "$ARCHIVE_MODE" -ne 1 ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] write archive manifest: $ARCHIVE_RUN_REL/manifest.txt"
        return 0
    fi

    mkdir -p "$ARCHIVE_RUN_DIR"
    {
        printf 'mode=archive\n'
        printf 'created_utc=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
        printf 'repo_root=%s\n' "$ROOT_DIR"
        printf 'archive_root=%s\n' "$ARCHIVE_RUN_REL"
        printf 'live_cleanup_targets=artifact bucket contents(preserving standard artifact dirs including bundles),logs,state/chat_history,state/events.jsonl,state/infer_history.jsonl,state/artifact_registry.jsonl,state/generated_image_provenance.jsonl,state/response_frames(compact_ledger,current_index,sidecar_snapshots),state/runtime_status.json,caches\n'
        printf 'protected_ghost_snapshot_paths=state/ghost_preferences.json,state/ghost_compiled_memory.json,state/ghost_compiled_memory.md,state/self_learning\n'
        printf 'protected_graph_rebase_registry_snapshot_path=state/graph_rebase/readiness_observations.jsonl\n'
        printf 'learning_retention_status=%s\n' "$LEARNING_RETENTION_STATUS"
        printf 'learning_retained_response_frame_sidecars=%s\n' "$LEARNING_RETAINED_SIDECAR_COUNT"
        printf 'learning_missing_response_frame_sidecars=%s\n' "$LEARNING_MISSING_SIDECAR_COUNT"
        printf 'learning_unsafe_response_frame_refs=%s\n' "$LEARNING_UNSAFE_REF_COUNT"
        printf 'graph_rebase_readiness_retention_status=%s\n' "$READINESS_RETENTION_STATUS"
        printf 'graph_rebase_readiness_selected_observations=%s\n' "$READINESS_SELECTED_OBSERVATION_COUNT"
        printf 'graph_rebase_readiness_settled_observations=%s\n' "$READINESS_SETTLED_OBSERVATION_COUNT"
        printf 'graph_rebase_readiness_registered_observations=%s\n' "$READINESS_REGISTERED_OBSERVATION_COUNT"
        printf 'graph_rebase_readiness_missing_settled_observations=%s\n' "$READINESS_MISSING_SETTLED_OBSERVATION_COUNT"
        printf 'graph_rebase_readiness_active_observations=%s\n' "$READINESS_ACTIVE_OBSERVATION_COUNT"
        printf 'graph_rebase_readiness_scan_errors=%s\n' "$READINESS_SCAN_ERROR_COUNT"
        printf 'graph_rebase_readiness_hydration_errors=%s\n' "$READINESS_HYDRATION_ERROR_COUNT"
        printf 'graph_rebase_readiness_registry_errors=%s\n' "$READINESS_REGISTRY_ERROR_COUNT"
        printf 'graph_rebase_readiness_errors=%s\n' "$READINESS_ERROR_COUNT"
        printf 'graph_rebase_readiness_appended_records=%s\n' "$READINESS_APPENDED_RECORD_COUNT"
        printf 'optional_flags=full_reset:%s reset_registry:%s forget_ghost:%s reset_llama_catalog:%s\n' "$FULL_RESET" "$RESET_REGISTRY" "$FORGET_GHOST" "$RESET_LLAMA_CATALOG"
    } > "$ARCHIVE_RUN_DIR/manifest.txt"
}

prepare_learning_retention() {
    local script="$ROOT_DIR/scripts/collect_self_learning_retention_roots.py"
    local manifest="$ROOT_DIR/state/self_learning/retention_manifest.json"
    local retained_dir="$ROOT_DIR/state/self_learning/retained_sidecars"
    local output
    local prefix=""

    if [[ "$DRY_RUN" -eq 1 ]]; then
        prefix="[dry-run] "
    fi

    if [[ ! -f "$script" ]]; then
        LEARNING_RETENTION_FAILED=1
        log "Warning: self-learning retention collector missing; preserving state/response_frames for safety."
        return 1
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        output="$(
            python3 "$script" \
                --self-learning-dir "$ROOT_DIR/state/self_learning" \
                --response-frames-dir "$ROOT_DIR/state/response_frames" \
                --retained-sidecars-dir "$retained_dir" \
                --dry-run \
                --verify \
                --shell-summary 2>&1
        )" || {
            LEARNING_RETENTION_FAILED=1
            log "Warning: self-learning retention collector failed; preserving state/response_frames for safety."
            log "$output"
            return 1
        }
    else
        output="$(
            python3 "$script" \
                --self-learning-dir "$ROOT_DIR/state/self_learning" \
                --response-frames-dir "$ROOT_DIR/state/response_frames" \
                --retained-sidecars-dir "$retained_dir" \
                --copy-retained \
                --write-manifest "$manifest" \
                --verify \
                --shell-summary 2>&1
        )" || {
            LEARNING_RETENTION_FAILED=1
            log "Warning: self-learning retention collector failed; preserving state/response_frames for safety."
            log "$output"
            return 1
        }
    fi

    LEARNING_RETENTION_STATUS="$(printf '%s\n' "$output" | awk -F= '/^status=/{print $2; exit}')"
    LEARNING_RETAINED_SIDECAR_COUNT="$(printf '%s\n' "$output" | awk -F= '/^retained_sidecar_count=/{print $2; exit}')"
    LEARNING_MISSING_SIDECAR_COUNT="$(printf '%s\n' "$output" | awk -F= '/^missing_sidecar_count=/{print $2; exit}')"
    LEARNING_UNSAFE_REF_COUNT="$(printf '%s\n' "$output" | awk -F= '/^external_or_unsafe_ref_count=/{print $2; exit}')"
    LEARNING_RETENTION_STATUS="${LEARNING_RETENTION_STATUS:-unknown}"
    LEARNING_RETAINED_SIDECAR_COUNT="${LEARNING_RETAINED_SIDECAR_COUNT:-0}"
    LEARNING_MISSING_SIDECAR_COUNT="${LEARNING_MISSING_SIDECAR_COUNT:-0}"
    LEARNING_UNSAFE_REF_COUNT="${LEARNING_UNSAFE_REF_COUNT:-0}"

    log "${prefix}preserve learning-retained response-frame sidecars: $LEARNING_RETAINED_SIDECAR_COUNT"
    log "${prefix}missing learning-retained sidecars: $LEARNING_MISSING_SIDECAR_COUNT"
    if [[ "$LEARNING_UNSAFE_REF_COUNT" != "0" ]]; then
        log "${prefix}unsafe learning-retained sidecar refs: $LEARNING_UNSAFE_REF_COUNT"
    fi
    if [[ "$LEARNING_RETENTION_STATUS" != "complete" && "$LEARNING_RETENTION_STATUS" != "empty" ]]; then
        LEARNING_RETENTION_FAILED=1
        log "Warning: self-learning retention is $LEARNING_RETENTION_STATUS; preserving state/response_frames for safety."
        return 1
    fi
    return 0
}

prepare_graph_rebase_readiness_retention() {
    local script="$ROOT_DIR/scripts/sync_graph_rebase_readiness_registry.py"
    local registry="$ROOT_DIR/state/graph_rebase/readiness_observations.jsonl"
    local frames_dir="$ROOT_DIR/state/response_frames"
    local output
    local prefix=""
    local mode_flag="--write"
    local value

    if [[ "$DRY_RUN" -eq 1 ]]; then
        prefix="[dry-run] "
        mode_flag="--check-only"
    fi

    if [[ ! -f "$script" ]]; then
        READINESS_RETENTION_FAILED=1
        READINESS_RETENTION_STATUS="missing_tool"
        log "Warning: graph-rebase readiness retention tool missing; preserving state/response_frames for safety."
        return 1
    fi

    output="$(
        python3 "$script" \
            --response-frames-dir "$frames_dir" \
            --registry "$registry" \
            "$mode_flag" \
            --require-all-settled \
            --shell-summary 2>&1
    )" || {
        READINESS_RETENTION_FAILED=1
        READINESS_RETENTION_STATUS="rejected"
        log "Warning: graph-rebase readiness retention preflight failed; preserving state/response_frames for safety."
        log "$output"
        return 1
    }

    READINESS_RETENTION_STATUS="$(printf '%s\n' "$output" | awk -F= '/^status=/{print $2; exit}')"
    READINESS_SELECTED_OBSERVATION_COUNT="$(printf '%s\n' "$output" | awk -F= '/^selected_observation_count=/{print $2; exit}')"
    READINESS_SETTLED_OBSERVATION_COUNT="$(printf '%s\n' "$output" | awk -F= '/^settled_observation_count=/{print $2; exit}')"
    READINESS_REGISTERED_OBSERVATION_COUNT="$(printf '%s\n' "$output" | awk -F= '/^registered_observation_count=/{print $2; exit}')"
    READINESS_MISSING_SETTLED_OBSERVATION_COUNT="$(printf '%s\n' "$output" | awk -F= '/^missing_settled_observation_count=/{print $2; exit}')"
    READINESS_ACTIVE_OBSERVATION_COUNT="$(printf '%s\n' "$output" | awk -F= '/^active_observation_count=/{print $2; exit}')"
    READINESS_SCAN_ERROR_COUNT="$(printf '%s\n' "$output" | awk -F= '/^scan_error_count=/{print $2; exit}')"
    READINESS_HYDRATION_ERROR_COUNT="$(printf '%s\n' "$output" | awk -F= '/^hydration_error_count=/{print $2; exit}')"
    READINESS_REGISTRY_ERROR_COUNT="$(printf '%s\n' "$output" | awk -F= '/^registry_error_count=/{print $2; exit}')"
    READINESS_ERROR_COUNT="$(printf '%s\n' "$output" | awk -F= '/^error_count=/{print $2; exit}')"
    READINESS_APPENDED_RECORD_COUNT="$(printf '%s\n' "$output" | awk -F= '/^appended_record_count=/{print $2; exit}')"

    READINESS_RETENTION_STATUS="${READINESS_RETENTION_STATUS:-unknown}"
    for value in \
        READINESS_SELECTED_OBSERVATION_COUNT \
        READINESS_SETTLED_OBSERVATION_COUNT \
        READINESS_REGISTERED_OBSERVATION_COUNT \
        READINESS_MISSING_SETTLED_OBSERVATION_COUNT \
        READINESS_ACTIVE_OBSERVATION_COUNT \
        READINESS_SCAN_ERROR_COUNT \
        READINESS_HYDRATION_ERROR_COUNT \
        READINESS_REGISTRY_ERROR_COUNT \
        READINESS_ERROR_COUNT \
        READINESS_APPENDED_RECORD_COUNT; do
        if [[ ! "${!value:-}" =~ ^[0-9]+$ ]]; then
            READINESS_RETENTION_FAILED=1
            READINESS_RETENTION_STATUS="invalid_summary"
            log "Warning: graph-rebase readiness retention returned an invalid shell summary; preserving state/response_frames for safety."
            log "$output"
            return 1
        fi
    done

    log "${prefix}graph-rebase readiness observations: $READINESS_SETTLED_OBSERVATION_COUNT settled / $READINESS_SELECTED_OBSERVATION_COUNT selected"
    log "${prefix}graph-rebase readiness registry: $READINESS_REGISTERED_OBSERVATION_COUNT registered, $READINESS_MISSING_SETTLED_OBSERVATION_COUNT pending append"
    if [[ "$READINESS_APPENDED_RECORD_COUNT" != "0" ]]; then
        log "${prefix}graph-rebase readiness records appended: $READINESS_APPENDED_RECORD_COUNT"
    fi

    if [[ "$READINESS_SELECTED_OBSERVATION_COUNT" != "$READINESS_SETTLED_OBSERVATION_COUNT" \
        || "$READINESS_ACTIVE_OBSERVATION_COUNT" != "0" \
        || "$READINESS_SCAN_ERROR_COUNT" != "0" \
        || "$READINESS_HYDRATION_ERROR_COUNT" != "0" \
        || "$READINESS_REGISTRY_ERROR_COUNT" != "0" \
        || "$READINESS_ERROR_COUNT" != "0" ]]; then
        READINESS_RETENTION_FAILED=1
        log "Warning: graph-rebase readiness evidence is active, incomplete, or invalid; preserving state/response_frames for safety."
        return 1
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        if [[ "$READINESS_RETENTION_STATUS" != "verified" ]]; then
            READINESS_RETENTION_FAILED=1
            log "Warning: graph-rebase readiness check status is $READINESS_RETENTION_STATUS; preserving state/response_frames for safety."
            return 1
        fi
    else
        if [[ "$READINESS_RETENTION_STATUS" != "written" && "$READINESS_RETENTION_STATUS" != "unchanged" ]]; then
            READINESS_RETENTION_FAILED=1
            log "Warning: graph-rebase readiness write status is $READINESS_RETENTION_STATUS; preserving state/response_frames for safety."
            return 1
        fi
        if [[ "$READINESS_MISSING_SETTLED_OBSERVATION_COUNT" != "0" \
            || "$READINESS_REGISTERED_OBSERVATION_COUNT" != "$READINESS_SETTLED_OBSERVATION_COUNT" ]]; then
            READINESS_RETENTION_FAILED=1
            log "Warning: not all settled graph-rebase readiness observations were registered; preserving state/response_frames for safety."
            return 1
        fi
    fi
    return 0
}

print_chrome_artifact_access_note() {
    log "Chrome/macOS artifact preview note:"
    log "- cleanup/archive preserves standard artifacts/ buckets; Chrome may need Desktop Folder or Full Disk Access re-granted before file:// bundle assets load."
    log "- symptom: local artifacts/bundles/.../index.html opens, but assets/css or assets/images fail with net::ERR_ACCESS_DENIED."
    log "- fix: quit Chrome, re-grant Chrome access in macOS Privacy & Security, or run clean/archive with --reset-chrome-file-access-prompt."
}

reset_chrome_file_access_prompt() {
    if [[ "$RESET_CHROME_FILE_ACCESS_PROMPT" -ne 1 ]]; then
        return 0
    fi

    log
    log "Resetting Chrome Desktop Folder permission prompt for local artifact previews..."
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "[dry-run] tccutil reset SystemPolicyDesktopFolder com.google.Chrome"
        return 0
    fi
    if [[ "$(uname -s 2>/dev/null || true)" != "Darwin" ]]; then
        log "Warning: --reset-chrome-file-access-prompt is macOS-only; skipping."
        return 0
    fi
    if ! command -v tccutil >/dev/null 2>&1; then
        log "Warning: tccutil not found; cannot reset Chrome file-access prompt."
        return 0
    fi
    if tccutil reset SystemPolicyDesktopFolder com.google.Chrome >/dev/null 2>&1; then
        log "- reset Chrome Desktop Folder permission; macOS should ask again on the next Chrome file:// bundle access."
    else
        log "Warning: could not reset Chrome Desktop Folder permission. You can run manually:"
        log "  tccutil reset SystemPolicyDesktopFolder com.google.Chrome"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --archiv|--archive)
            ARCHIVE_MODE=1
            ;;
        --full|--empty-state)
            FULL_RESET=1
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        --reset-registry)
            RESET_REGISTRY=1
            ;;
        --forget-ghost)
            FORGET_GHOST=1
            ;;
        --reset-llama-catalog)
            RESET_LLAMA_CATALOG=1
            ;;
        --reset-chrome-file-access-prompt)
            RESET_CHROME_FILE_ACCESS_PROMPT=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if [[ "$FULL_RESET" -eq 1 ]]; then
    RESET_REGISTRY=1
    RESET_LLAMA_CATALOG=1
    if [[ "$ARCHIVE_MODE" -eq 0 ]]; then
        FORGET_GHOST=1
    fi
fi

if [[ "$ARCHIVE_MODE" -eq 1 ]]; then
    ARCHIVE_RUN_REL="$ARCHIVE_BASE_REL/$(timestamp_utc)"
    ARCHIVE_RUN_DIR="$ROOT_DIR/$ARCHIVE_RUN_REL"
fi

log "=== Ollmo Repo Reset ==="
log "repo: $ROOT_DIR"
if [[ "$ARCHIVE_MODE" -eq 1 ]]; then
    log "mode: archive"
    log "archive: $ARCHIVE_RUN_REL"
elif [[ "$DRY_RUN" -eq 1 ]]; then
    log "mode: dry-run"
fi
if [[ "$FULL_RESET" -eq 1 ]]; then
    log "full reset: enabled"
fi
if [[ "$DRY_RUN" -eq 1 && "$ARCHIVE_MODE" -eq 1 ]]; then
    log "note: dry-run only, archive is not created"
fi

log
log "1. Stopping repo-local listeners on standard Ollmo ports..."
stop_listener_on_port 5001
stop_listener_on_port 11434
for port in $(seq 11435 11550); do
    stop_listener_on_port "$port"
done

log
log "Preparing self-learning retained evidence..."
prepare_learning_retention || true

log
log "Preparing durable graph-rebase readiness evidence..."
prepare_graph_rebase_readiness_retention || true

if [[ "$ARCHIVE_MODE" -eq 1 ]]; then
    log
    log "2. Archiving useful runtime/generated ballast into $ARCHIVE_RUN_REL ..."
    archive_artifacts_preserving_base_dirs
    archive_dir_contents "logs"
    archive_dir_contents "state/chat_history"
    archive_path "state/events.jsonl"
    archive_path "state/infer_history.jsonl"
    archive_path "state/artifact_registry.jsonl"
    archive_path "state/generated_image_provenance.jsonl"
    if [[ "$LEARNING_RETENTION_FAILED" -eq 1 || "$READINESS_RETENTION_FAILED" -eq 1 ]]; then
        log "Warning: skipping archive of state/response_frames because an evidence-retention preflight failed."
    else
        archive_dir_contents "state/response_frames"
    fi
    archive_path "state/runtime_status.json"

    snapshot_path "state/ghost_preferences.json"
    snapshot_path "state/ghost_compiled_memory.json"
    snapshot_path "state/ghost_compiled_memory.md"
    snapshot_path "state/self_learning"
    snapshot_path "state/graph_rebase/readiness_observations.jsonl"

    if [[ "$RESET_LLAMA_CATALOG" -eq 1 ]]; then
        archive_path "state/llama_cpp_catalog.json"
    fi

    if [[ "$RESET_REGISTRY" -eq 1 ]]; then
        archive_path "model_ports.json"
    fi

    write_archive_manifest
    STEP_LABEL="3"
else
    STEP_LABEL="2"
fi

log
log "${STEP_LABEL}. Removing repo-local generated/runtime ballast..."
remove_artifacts_preserving_base_dirs
remove_dir_contents "logs"
remove_path "outputs"
remove_dir_contents "state/chat_history"
if [[ "$LEARNING_RETENTION_FAILED" -eq 1 || "$READINESS_RETENTION_FAILED" -eq 1 ]]; then
    log "Warning: preserving state/response_frames because an evidence-retention preflight failed."
else
    remove_dir_contents "state/response_frames"
fi

remove_path "state/events.jsonl"
remove_path "state/infer_history.jsonl"
remove_path "state/artifact_registry.jsonl"
remove_path "state/generated_image_provenance.jsonl"
remove_path "state/runtime_status.json"

remove_path ".pytest_cache"
remove_path ".coverage"
remove_path "htmlcov"
remove_path ".mypy_cache"
remove_path ".ruff_cache"

remove_cache_dirs "__pycache__"
remove_cache_files "*.pyc"
remove_cache_files "*.pyo"
remove_cache_files ".DS_Store"

if [[ "$FORGET_GHOST" -eq 1 ]]; then
    log
    log "Removing Ghost preference state and compiled-memory residue..."
    remove_path "state/ghost_preferences.json"
    remove_path "state/ghost_compiled_memory.json"
    remove_path "state/ghost_compiled_memory.md"
fi

if [[ "$RESET_LLAMA_CATALOG" -eq 1 ]]; then
    log
    log "Removing llama.cpp catalog state..."
    remove_path "state/llama_cpp_catalog.json"
fi

if [[ "$RESET_REGISTRY" -eq 1 ]]; then
    log
    log "Resetting repo-local model registry..."
    write_json_array "model_ports.json"
fi

log
log "Ensuring standard runtime dirs exist (fallback for missing dirs)..."
ensure_artifact_base_dirs
ensure_dir "logs"
ensure_dir "state/chat_history"
ensure_dir "state/response_frames/snapshots"

log
log "Final cache/Finder sweep..."
remove_path ".pytest_cache"
remove_cache_dirs "__pycache__"
remove_cache_files "*.pyc"
remove_cache_files "*.pyo"
remove_cache_files ".DS_Store"

log
log "Preserved/reset summary:"
if [[ "$RESET_REGISTRY" -eq 1 ]]; then
    log "- model_ports.json reset to []"
else
    log "- model_ports.json preserved"
fi
if [[ "$RESET_LLAMA_CATALOG" -eq 1 ]]; then
    log "- state/llama_cpp_catalog.json removed"
else
    log "- state/llama_cpp_catalog.json preserved"
fi
if [[ "$FORGET_GHOST" -eq 1 ]]; then
    log "- state/ghost_preferences.json removed"
else
    log "- state/ghost_preferences.json preserved"
fi
if [[ "$ARCHIVE_MODE" -eq 1 ]]; then
    log "- protected Ghost state snapshotted when present; preferences/compiled memory preserved unless --forget-ghost was explicit"
fi
if [[ "$ARCHIVE_MODE" -eq 1 ]]; then
    log "- archived runtime/generated ballast under $ARCHIVE_RUN_REL"
fi
log "- learning-retained response-frame sidecars: $LEARNING_RETAINED_SIDECAR_COUNT retained, $LEARNING_MISSING_SIDECAR_COUNT missing"
log "- graph-rebase readiness registry preserved: $READINESS_REGISTERED_OBSERVATION_COUNT / $READINESS_SETTLED_OBSERVATION_COUNT settled observations registered"
if [[ "$READINESS_RETENTION_FAILED" -eq 1 ]]; then
    log "- state/response_frames preserved because graph-rebase readiness retention did not pass"
fi
print_chrome_artifact_access_note
reset_chrome_file_access_prompt

if [[ "$DRY_RUN" -eq 0 ]]; then
    log
    log "Post-reset status:"
    "$ROOT_DIR/ollmo" status || true
    log
    log "Repo-local reset complete."
fi
