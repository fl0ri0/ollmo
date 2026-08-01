#!/usr/bin/env python3
"""
Fire simultaneous chat-completion requests against one or more local providers.

Usage examples:

    # Single target (legacy behavior)
    python scripts/probe_provider_concurrency.py --port 11502 --requests 2

    # Two targets at once (shared barrier so both ports fire concurrently)
    python scripts/probe_provider_concurrency.py --port 11435,11502 --requests 3 \
        --prompt "Explain concurrency tests in depth."

    # Mix of instance IDs and ports
    python scripts/probe_provider_concurrency.py --instance gpt-oss:latest-1 \
        --port 11502 --requests 2 --prompt "Describe concurrency probe."

    # Automatically target every non-agent entry in model_ports.json
    python scripts/probe_provider_concurrency.py --all --requests 3 \
        --prompt "Write a 2000-word stress test."

The script reads `model_ports.json`, selects every requested entry, and issues
multiple HTTP POST /v1/chat/completions calls in parallel using stdlib clients.
Timing data per target helps confirm whether each backend handles requests
concurrently or queues them, even while other ports are under load.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

MODEL_PORTS_PATH = Path("model_ports.json")


@dataclass
class TargetEntry:
    instance_id: str
    model: str
    port: int
    backend: Optional[str]


@dataclass
class ProbeResult:
    target_label: str
    label: str
    started: float
    first_byte: Optional[float]
    finished: Optional[float]
    status: Optional[int]
    snippet: str
    error: Optional[str]

    @property
    def duration(self) -> Optional[float]:
        if self.finished is None or self.started is None:
            return None
        return self.finished - self.started


def load_inventory() -> List[TargetEntry]:
    if not MODEL_PORTS_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PORTS_PATH}")
    data = json.loads(MODEL_PORTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("model_ports.json must contain a list")
    entries: List[TargetEntry] = []
    for row in data:
        if row.get("agent"):
            continue
        try:
            entries.append(
                TargetEntry(
                    instance_id=str(row["instance_id"]),
                    model=str(row["model"]),
                    port=int(row["port"]),
                    backend=row.get("backend"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid entry in model_ports.json: {row!r}") from exc
    if not entries:
        raise ValueError("No non-agent entries found in model_ports.json")
    return entries


def select_targets(
    inventory: List[TargetEntry],
    *,
    ports: List[int],
    instances: List[str],
) -> List[TargetEntry]:
    selected: List[TargetEntry] = []
    seen_keys = set()

    def add_entry(entry: TargetEntry) -> None:
        key = (entry.instance_id.lower(), entry.port)
        if key not in seen_keys:
            selected.append(entry)
            seen_keys.add(key)

    if not ports and not instances:
        return [inventory[0]]

    for port in ports:
        entry = next((row for row in inventory if row.port == port), None)
        if entry is None:
            raise ValueError(f"No provider found for port={port}. Check model_ports.json.")
        add_entry(entry)

    for instance in instances:
        normalized = instance.lower()
        entry = next(
            (row for row in inventory if row.instance_id.lower() == normalized),
            None,
        )
        if entry is None:
            raise ValueError(
                f"No provider found for instance_id={instance!r}. Check model_ports.json."
            )
        add_entry(entry)

    if not selected:
        raise ValueError("No providers selected; check --port/--instance values.")
    return selected


def build_payload(model_name: str, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    return {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are measuring backend concurrency; respond succinctly.",
            },
            {
                "role": "user",
                "content": prompt.strip() or "Describe how you process simultaneous requests.",
            },
        ],
    }


def extract_snippet(body: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:120].replace("\n", " ")
    choices = payload.get("choices") or []
    if not choices:
        return body[:120].replace("\n", " ")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return content.strip().replace("\n", " ")[:160]


def post_once(
    target_label: str,
    label: str,
    entry: TargetEntry,
    payload: Dict[str, Any],
    *,
    barrier: threading.Barrier,
    timeout: int,
) -> ProbeResult:
    started = time.perf_counter()
    first_byte: Optional[float] = None
    finished: Optional[float] = None
    status: Optional[int] = None
    snippet = ""
    error: Optional[str] = None

    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        error = "Barrier broken before request start"
        return ProbeResult(
            target_label, label, started, first_byte, finished, status, snippet, error
        )

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    conn = http.client.HTTPConnection("127.0.0.1", entry.port, timeout=timeout)

    try:
        conn.request("POST", "/v1/chat/completions", body, headers)
        response = conn.getresponse()
        status = response.status
        first_byte = time.perf_counter()
        raw = response.read().decode("utf-8", errors="replace")
        finished = time.perf_counter()
        snippet = extract_snippet(raw)
    except Exception as exc:  # noqa: BLE001 - need to surface networking issues.
        error = str(exc)
        finished = time.perf_counter()
    finally:
        conn.close()

    return ProbeResult(
        target_label, label, started, first_byte, finished, status, snippet, error
    )


def summarize_grouped(results_by_target: Dict[str, List[ProbeResult]]) -> None:
    if not results_by_target:
        print("No results collected.")
        return

    for target_label, results in results_by_target.items():
        print(f"\n=== Target: {target_label} ===")
        if not results:
            print("No results for this target.")
            continue

        print(f"{'req':<16} {'status':<8} {'duration(s)':<12} {'first_byte(s)':<14} snippet/error")
        for res in sorted(results, key=lambda r: r.label):
            dur = f"{res.duration:.2f}" if res.duration is not None else "n/a"
            first = (
                f"{(res.first_byte - res.started):.2f}"
                if res.first_byte is not None
                else "n/a"
            )
            status_txt = res.status if res.status is not None else "--"
            snippet = f"ERROR: {res.error}" if res.error else (res.snippet or "(empty)")
            print(f"{res.label:<16} {status_txt!s:<8} {dur:<12} {first:<14} {snippet}")

        successful = [res for res in results if res.duration is not None]
        if not successful:
            continue
        start_min = min(res.started for res in successful)
        end_max = max(res.finished for res in successful if res.finished is not None)
        total_wall = end_max - start_min if end_max and start_min else None
        sum_durations = sum(res.duration or 0.0 for res in successful)

        print("Summary:")
        if total_wall is not None:
            print(f"- Wall-clock span: {total_wall:.2f}s")
        print(f"- Sum of durations: {sum_durations:.2f}s")
        if total_wall is not None:
            if total_wall < sum_durations * 0.75:
                verdict = (
                    "Requests likely overlapped (wall time << sum of per-request durations)."
                )
            elif total_wall > sum_durations * 0.9:
                verdict = (
                    "Requests likely serialized (wall time ~= sum of per-request durations)."
                )
            else:
                verdict = "Mixed behavior detected."
            print(f"- Verdict: {verdict}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether a local provider port handles concurrent chat completions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        help="Comma-separated provider ports to target (overrides --instance).",
    )
    parser.add_argument(
        "--instance",
        help="Comma-separated instance_ids from model_ports.json.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Target every non-agent entry from model_ports.json (overrides --port/--instance).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=2,
        help="Number of simultaneous requests to fire.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe your behavior when multiple requests arrive at once.",
        help="User prompt to send.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature value for the request payload.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="max_tokens for the chat completion request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request HTTP timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.requests < 1:
        print("--requests must be >= 1", file=sys.stderr)
        return 2

    try:
        inventory = load_inventory()
        if args.all:
            targets = list(inventory)
        else:
            ports = (
                [int(item) for item in args.port.split(",") if item.strip()]
                if args.port
                else []
            )
            instances = (
                [item.strip() for item in args.instance.split(",") if item.strip()]
                if args.instance
                else []
            )
            targets = select_targets(inventory, ports=ports, instances=instances)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    total_threads = args.requests * len(targets)
    if total_threads == 0:
        print("[error] No requests scheduled; check --requests/targets.", file=sys.stderr)
        return 2

    barrier = threading.Barrier(total_threads)
    results_by_target: Dict[str, List[ProbeResult]] = {entry.instance_id: [] for entry in targets}

    print(
        "Targets:"
        + "".join(
            f"\n- {entry.instance_id} ({entry.model}) on port {entry.port} "
            f"[backend={entry.backend or 'unknown'}]"
            for entry in targets
        )
    )
    print(
        f"\nFiring {args.requests} concurrent request(s) per target "
        f"(total threads={total_threads}) with timeout={args.timeout}s. "
        "All requests share one barrier so different ports ignite simultaneously."
    )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as executor:
            futures = [
                executor.submit(
                    post_once,
                    entry.instance_id,
                    f"{entry.instance_id}-req-{idx+1}",
                    entry,
                    build_payload(entry.model, args.prompt, args.temperature, args.max_tokens),
                    barrier=barrier,
                    timeout=args.timeout,
                )
                for entry in targets
                for idx in range(args.requests)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results_by_target.setdefault(result.target_label, []).append(result)
    except KeyboardInterrupt:
        print("\nInterrupted by user; partial results:")
    finally:
        summarize_grouped(results_by_target)

    exit_code = 0
    for per_target in results_by_target.values():
        for res in per_target:
            if res.error or (res.status or 0) >= 400:
                exit_code = 3
                break
        if exit_code:
            break
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
