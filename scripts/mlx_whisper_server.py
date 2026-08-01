#!/usr/bin/env python3
"""Minimal HTTP server for MLX Whisper transcription."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _load_request_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def build_handler(model_path: str):
    class WhisperHandler(BaseHTTPRequestHandler):
        server_version = "mlx-whisper-server/0.1"

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            print(f"[mlx-whisper] {self.address_string()} - {format % args}")

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/healthz":
                _json_response(
                    self,
                    {
                        "ok": True,
                        "capability": "speech_to_text",
                        "backend": "mlx",
                        "model_path": model_path,
                    },
                    status=200,
                )
                return
            _json_response(self, {"error": "Not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") not in {"/v1/audio/transcriptions", "/api/transcribe"}:
                _json_response(self, {"error": "Not found"}, status=404)
                return

            try:
                payload = _load_request_json(self)
            except json.JSONDecodeError as exc:
                _json_response(self, {"error": f"Invalid JSON body: {exc}"}, status=400)
                return

            audio_path = str(payload.get("audio_path") or "").strip()
            if not audio_path:
                _json_response(self, {"error": "Parameter 'audio_path' fehlt."}, status=400)
                return

            if not Path(audio_path).exists():
                _json_response(self, {"error": f"Audio file was not found: {audio_path}"}, status=400)
                return

            language = payload.get("language")
            if isinstance(language, str):
                language = language.strip() or None
            else:
                language = None

            task = payload.get("task")
            if not isinstance(task, str) or not task.strip():
                task = "transcribe"

            word_timestamps = bool(payload.get("word_timestamps", False))

            try:
                from mlx_whisper import transcribe as whisper_transcribe

                result = whisper_transcribe(
                    audio_path,
                    path_or_hf_repo=model_path,
                    verbose=False,
                    language=language,
                    task=task,
                    word_timestamps=word_timestamps,
                )
            except Exception as exc:  # noqa: BLE001
                _json_response(self, {"error": str(exc)}, status=500)
                return

            _json_response(
                self,
                {
                    "text": result.get("text", ""),
                    "segments": result.get("segments", []),
                    "language": result.get("language"),
                    "model_path": model_path,
                },
                status=200,
            )

    return WhisperHandler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    handler_cls = build_handler(args.model_path)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(
        f"[mlx-whisper] serving on http://{args.host}:{args.port} "
        f"with model path: {args.model_path}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
