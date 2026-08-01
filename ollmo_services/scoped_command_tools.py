"""Scoped internal command runner for Ollmo control-plane loops."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ollmo_services.scoped_file_tools import expand_scoped_roots, resolve_scoped_path

DEFAULT_COMMAND_ROOTS = (Path('.'),)
DEFAULT_ENV_KEYS = ('HOME', 'LANG', 'LC_ALL', 'PATH', 'PYTHONPATH', 'VIRTUAL_ENV')
DEFAULT_TIMEOUT_SEC = 30
DEFAULT_MAX_OUTPUT_BYTES = 200_000


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    if isinstance(argv, (str, bytes)):
        raise TypeError('argv must be a sequence of arguments, not a shell string')
    normalized = [str(item) for item in argv if str(item)]
    if not normalized:
        raise ValueError('argv must not be empty')
    return normalized


def _prefix_allowed(argv: list[str], allowed_prefixes: Optional[Iterable[Sequence[str]]]) -> bool:
    prefixes = list(allowed_prefixes or [])
    if not prefixes:
        return True
    for raw_prefix in prefixes:
        prefix = [str(item) for item in raw_prefix if str(item)]
        if prefix and argv[: len(prefix)] == prefix:
            return True
    return False


def _resolve_executable(argv: list[str], *, cwd: Path, allowed_roots: set[Path]) -> list[str]:
    executable = Path(argv[0])
    if not executable.is_absolute() and executable.parent == Path('.'):
        return argv
    resolved = executable if executable.is_absolute() else cwd / executable
    resolved = resolved.resolve(strict=False)
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        raise ValueError(f'executable path is outside allowed command roots: {resolved}')
    return [str(resolved), *argv[1:]]


def _build_env(
    *,
    env: Optional[dict[str, str]] = None,
    inherit_env_keys: Iterable[str] = DEFAULT_ENV_KEYS,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key in inherit_env_keys:
        token = str(key or '').strip()
        if token and token in os.environ:
            payload[token] = os.environ[token]
    for key, value in (env or {}).items():
        token = str(key or '').strip()
        if token:
            payload[token] = str(value)
    return payload


def _decode_capped(value: bytes | str | None, *, max_bytes: int) -> tuple[str, bool]:
    if value is None:
        return '', False
    raw = value.encode('utf-8', errors='replace') if isinstance(value, str) else value
    limit = max(1, int(max_bytes or DEFAULT_MAX_OUTPUT_BYTES))
    capped = raw[:limit]
    return capped.decode('utf-8', errors='replace'), len(raw) > len(capped)


def run_scoped_command(
    argv: Sequence[str],
    *,
    cwd: Path | str = '.',
    allowed_cwd_roots: Optional[Iterable[Path | str]] = None,
    allowed_command_prefixes: Optional[Iterable[Sequence[str]]] = None,
    repo_root: Path | str | None = None,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    env: Optional[dict[str, str]] = None,
    inherit_env_keys: Iterable[str] = DEFAULT_ENV_KEYS,
) -> dict:
    command = _normalize_argv(argv)
    if int(timeout_sec or 0) <= 0:
        raise ValueError('timeout_sec must be positive')
    if not _prefix_allowed(command, allowed_command_prefixes):
        raise ValueError(f'command is not allowed: {command[0]}')

    roots = expand_scoped_roots(allowed_cwd_roots or DEFAULT_COMMAND_ROOTS, repo_root=repo_root)
    resolved_cwd = resolve_scoped_path(
        cwd,
        allowed_roots=roots,
        repo_root=repo_root,
        must_exist=True,
    )
    if not resolved_cwd.is_dir():
        raise NotADirectoryError(f'cwd is not a directory: {resolved_cwd}')
    resolved_command = _resolve_executable(command, cwd=resolved_cwd, allowed_roots=roots)

    try:
        completed = subprocess.run(
            resolved_command,
            cwd=str(resolved_cwd),
            env=_build_env(env=env, inherit_env_keys=inherit_env_keys),
            capture_output=True,
            check=False,
            shell=False,
            timeout=int(timeout_sec),
        )
        stdout, stdout_truncated = _decode_capped(completed.stdout, max_bytes=max_output_bytes)
        stderr, stderr_truncated = _decode_capped(completed.stderr, max_bytes=max_output_bytes)
        return {
            'operation': 'run',
            'argv': resolved_command,
            'cwd': str(resolved_cwd),
            'returncode': completed.returncode,
            'timed_out': False,
            'stdout': stdout,
            'stderr': stderr,
            'stdout_truncated': stdout_truncated,
            'stderr_truncated': stderr_truncated,
        }
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _decode_capped(exc.stdout, max_bytes=max_output_bytes)
        stderr, stderr_truncated = _decode_capped(exc.stderr, max_bytes=max_output_bytes)
        return {
            'operation': 'run',
            'argv': resolved_command,
            'cwd': str(resolved_cwd),
            'returncode': None,
            'timed_out': True,
            'stdout': stdout,
            'stderr': stderr,
            'stdout_truncated': stdout_truncated,
            'stderr_truncated': stderr_truncated,
        }
