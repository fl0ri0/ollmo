from __future__ import annotations

from typing import Any


def _ollama_base_url(port: int) -> str:
    return f"http://127.0.0.1:{int(port)}"


def _requests_module():
    import requests

    return requests


def fetch_ollama_show(port: int, model: str, *, timeout: int = 10) -> dict[str, Any]:
    requests = _requests_module()
    response = requests.post(
        f"{_ollama_base_url(port)}/api/show",
        json={"model": model},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def fetch_ollama_ps(port: int, *, timeout: int = 5) -> dict[str, Any]:
    requests = _requests_module()
    response = requests.get(
        f"{_ollama_base_url(port)}/api/ps",
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def ollama_chat(port: int, model: str, messages: list[dict], timeout=90) -> str:
    """
    Call a specific Ollama port with a chat payload and return plain text.
    Accepts native /api/chat output as well as OpenAI-compatible adapters.
    """
    url = f"{_ollama_base_url(port)}/api/chat"
    requests = _requests_module()
    r = requests.post(url, json={"model": model, "messages": messages, "stream": False}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # Native Ollama /api/chat:
    text = (data.get("message") or {}).get("content", "")
    if text:
        return text
    # Some adapters return {"response": "..."}
    return data.get("response", "")
