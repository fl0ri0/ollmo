#!/usr/bin/env python3
"""Submit a prompt batch to running Flux instances via the local Ollmo API.

This script auto-discovers ready image_generation instances from model_ports.json,
then distributes prompts across them with one sequential worker per instance.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ollmo_core.transports import ARTIFACT_OUTPUTS_MANIFESTS_DIR

DEFAULT_MODEL_PORTS_PATH = ROOT / "model_ports.json"
DEFAULT_OUTPUT_DIR = ROOT / ARTIFACT_OUTPUTS_MANIFESTS_DIR


def load_prompts(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array of prompt strings.")
    prompts = [str(item).strip() for item in payload if str(item).strip()]
    if not prompts:
        raise ValueError(f"{path} does not contain any non-empty prompts.")
    return prompts


def discover_instances(model_ports_path: Path, limit: int) -> list[str]:
    payload = json.loads(model_ports_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{model_ports_path} must contain a JSON array.")
    ready_instances = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if str(item.get("capability") or "").strip() != "image_generation":
            continue
        instance_id = str(item.get("instance_id") or "").strip()
        if instance_id:
            ready_instances.append(instance_id)
    if limit > 0:
        ready_instances = ready_instances[:limit]
    if not ready_instances:
        raise ValueError("No image_generation instances found in model_ports.json.")
    return ready_instances


def build_jobs(prompts: list[str], instances: list[str]) -> dict[str, list[dict[str, Any]]]:
    jobs: dict[str, list[dict[str, Any]]] = {instance_id: [] for instance_id in instances}
    for index, prompt in enumerate(prompts, start=1):
        instance_id = instances[(index - 1) % len(instances)]
        jobs[instance_id].append(
            {
                "index": index,
                "prompt": prompt,
                "instance_id": instance_id,
            }
        )
    return jobs


def worker(
    *,
    instance_id: str,
    jobs: list[dict[str, Any]],
    base_url: str,
    timeout_sec: int,
    results: list[dict[str, Any]],
    errors: queue.Queue,
    lock: threading.Lock,
) -> None:
    endpoint = f"{base_url.rstrip('/')}/api/responses"
    session = requests.Session()
    for job in jobs:
        started_at = time.time()
        payload = {
            "instance_id": instance_id,
            "capability": "image_generation",
            "prompt": job["prompt"],
            "infer_timeout_sec": timeout_sec,
        }
        try:
            response = session.post(endpoint, json=payload, timeout=max(timeout_sec + 60, 120))
            response.raise_for_status()
            body = response.json()
            item = {
                "index": job["index"],
                "instance_id": instance_id,
                "prompt": job["prompt"],
                "saved_image_path": body.get("saved_image_path"),
                "image_data_url": body.get("image_data_url"),
                "mode": body.get("mode"),
                "content": body.get("content"),
                "latency_sec": round(time.time() - started_at, 3),
            }
            with lock:
                results.append(item)
            print(
                f"[{instance_id}] {job['index']:02d} OK "
                f"({item['latency_sec']}s) -> {item['saved_image_path'] or 'inline image'}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            errors.put(
                {
                    "index": job["index"],
                    "instance_id": instance_id,
                    "prompt": job["prompt"],
                    "error": str(exc),
                }
            )
            print(f"[{instance_id}] {job['index']:02d} FAIL -> {exc}", flush=True)


def write_manifest(
    *,
    prompts_path: Path,
    instances: list[str],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    manifest_path = output_dir / f"flux_batch_{stamp}.json"
    payload = {
        "created_at": stamp,
        "prompts_path": str(prompts_path),
        "instances": instances,
        "result_count": len(results),
        "failure_count": len(failures),
        "results": sorted(results, key=lambda item: int(item["index"])),
        "failures": sorted(failures, key=lambda item: int(item["index"])),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLMO_BASE_URL", "http://127.0.0.1:5001"),
        help="Local Ollmo base URL (default: http://127.0.0.1:5001).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Path to a JSON array of prompts.",
    )
    parser.add_argument(
        "--model-ports",
        type=Path,
        default=DEFAULT_MODEL_PORTS_PATH,
        help=f"Path to model_ports.json (default: {DEFAULT_MODEL_PORTS_PATH}).",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=[],
        help="Explicit instance IDs to use. If omitted, the script auto-discovers image_generation instances.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=4,
        help="Maximum number of discovered image_generation instances to use.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=600,
        help="Backend infer timeout per image request.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the result manifest (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    instances = args.instances or discover_instances(args.model_ports, args.max_instances)
    jobs_by_instance = build_jobs(prompts, instances)

    print(f"Using {len(instances)} Flux instance(s): {', '.join(instances)}", flush=True)
    print(f"Submitting {len(prompts)} prompts from {args.prompts}", flush=True)

    results: list[dict[str, Any]] = []
    failures_q: queue.Queue = queue.Queue()
    lock = threading.Lock()
    threads = []
    for instance_id in instances:
        jobs = jobs_by_instance.get(instance_id) or []
        if not jobs:
            continue
        thread = threading.Thread(
            target=worker,
            kwargs={
                "instance_id": instance_id,
                "jobs": jobs,
                "base_url": args.base_url,
                "timeout_sec": args.timeout_sec,
                "results": results,
                "errors": failures_q,
                "lock": lock,
            },
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    failures = []
    while not failures_q.empty():
        failures.append(failures_q.get())

    manifest_path = write_manifest(
        prompts_path=args.prompts,
        instances=instances,
        results=results,
        failures=failures,
        output_dir=args.output_dir,
    )
    print(f"Manifest written to: {manifest_path}", flush=True)

    if failures:
        print(f"{len(failures)} prompt(s) failed.", flush=True)
        return 1

    print(f"Completed {len(results)} image request(s) successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
