import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ollmo_runtime.registry import (
    read_registry_entries,
    write_registry_entries,
    is_port_listening as registry_is_port_listening,
)

CONFIG_PATH = Path("model_ports.json")
AGENT_FLAG = "agent"
DEFAULT_PORT_RANGE = range(6200, 6300)


def load_entries() -> List[Dict[str, Any]]:
    return read_registry_entries(CONFIG_PATH)


def save_entries(entries: List[Dict[str, Any]]) -> None:
    write_registry_entries(
        entries,
        path=CONFIG_PATH,
        preserve_agents=False,
        sync_external=False,
    )


def is_port_in_use(port: int) -> bool:
    return registry_is_port_listening(port, "127.0.0.1")


def find_free_port(preferred: Optional[int], taken: set[int]) -> int:
    if preferred and preferred not in taken and not is_port_in_use(preferred):
        return preferred
    for candidate in DEFAULT_PORT_RANGE:
        if candidate in taken:
            continue
        if not is_port_in_use(candidate):
            return candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ensure_agent_entry(role: str, preferred_port: Optional[int] = None) -> Dict[str, Any]:
    entries = load_entries()
    agent_entry = None
    taken_ports = {int(entry.get("port")) for entry in entries if entry.get("port")}

    for entry in entries:
        if entry.get(AGENT_FLAG) and entry.get("role") == role:
            agent_entry = entry
            break

    if agent_entry:
        port = int(agent_entry.get("port") or 0)
        if port:
            agent_entry["pid"] = None
            save_entries(entries)
            return agent_entry
    else:
        agent_entry = {
            "instance_id": f"agent-{role}",
            "model": f"agent:{role}",
            "role": role,
            AGENT_FLAG: True,
        }
        entries.append(agent_entry)

    port = find_free_port(preferred_port, taken_ports)
    agent_entry["port"] = int(port)
    agent_entry["pid"] = None
    save_entries(entries)
    return agent_entry


def update_agent_pid(role: str, pid: Optional[int]) -> None:
    entries = load_entries()
    updated = False
    for entry in entries:
        if entry.get(AGENT_FLAG) and entry.get("role") == role:
            entry["pid"] = int(pid) if pid else None
            entry["ts"] = int(time.time())
            updated = True
            break
    if updated:
        save_entries(entries)


def remove_agent_entry(role: str) -> None:
    entries = load_entries()
    filtered = [entry for entry in entries if not (entry.get(AGENT_FLAG) and entry.get("role") == role)]
    if len(filtered) != len(entries):
        save_entries(filtered)


def list_agent_entries() -> List[Dict[str, Any]]:
    return [entry for entry in load_entries() if entry.get(AGENT_FLAG)]


def stop_agent(role: str, timeout: float = 5.0) -> bool:
    entries = load_entries()
    target = next((entry for entry in entries if entry.get(AGENT_FLAG) and entry.get("role") == role), None)
    if not target:
        return False
    pid = target.get("pid")
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pid = None
        except PermissionError:
            pass
    port = target.get("port")
    if port:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not is_port_in_use(int(port)):
                break
            time.sleep(0.2)
    remove_agent_entry(role)
    return True


def stop_all_agents() -> None:
    for entry in list_agent_entries():
        stop_agent(entry.get("role", ""), timeout=5.0)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Manage agent registrations in model_ports.json")
    sub = parser.add_subparsers(dest="command", required=True)

    reserve = sub.add_parser("reserve", help="Ensure agent entry exists and return assigned port")
    reserve.add_argument("--role", required=True)
    reserve.add_argument("--prefer", type=int, default=None)

    update = sub.add_parser("update", help="Update stored PID for an agent role")
    update.add_argument("--role", required=True)
    update.add_argument("--pid", type=int)

    remove = sub.add_parser("remove", help="Remove agent entry")
    remove.add_argument("--role", required=True)

    list_cmd = sub.add_parser("list", help="List agent entries", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    list_cmd.add_argument("--pretty", action="store_true")

    stop_cmd = sub.add_parser("stop", help="Stop a specific agent by role")
    stop_cmd.add_argument("--role", required=True)

    sub.add_parser("stop-all", help="Stop all registered agents")

    args = parser.parse_args(argv)

    if args.command == "reserve":
        entry = ensure_agent_entry(args.role, args.prefer)
        print(entry.get("port"))
        return 0
    if args.command == "update":
        update_agent_pid(args.role, args.pid)
        return 0
    if args.command == "remove":
        remove_agent_entry(args.role)
        return 0
    if args.command == "list":
        entries = list_agent_entries()
        if args.pretty:
            import json
            print(json.dumps(entries, indent=2))
        else:
            import json
            print(json.dumps(entries))
        return 0
    if args.command == "stop":
        stop_agent(args.role)
        return 0
    if args.command == "stop-all":
        stop_all_agents()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
