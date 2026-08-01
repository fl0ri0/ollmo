from __future__ import annotations

import base64
import hashlib
import json
import struct
from pathlib import Path


PNG_BYTES_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

TEXT_ARTIFACT_CONTENT = (
    "# Runtime Artifact\n\n"
    "This markdown file is deterministic fake-backend output. "
    "The saved file, registry record, and response frame are the truth.\n"
)

TRANSCRIPT_TEXT = "Deterministic local transcript from fake speech input."

VISION_RESULT = {
    "description": "A deterministic one-pixel PNG fixture used by the fake vision backend.",
    "ocr": ["FAKE OCR"],
    "labels": ["fixture", "png", "deterministic"],
}


def tiny_png_bytes() -> bytes:
    return base64.b64decode(PNG_BYTES_BASE64)


def tiny_wav_bytes() -> bytes:
    sample_rate = 8000
    sample_count = int(sample_rate * 1.5)
    period = max(2, sample_rate // 220)
    samples = [
        9000 if (index % period) < (period // 2) else -9000
        for index in range(sample_count)
    ]
    pcm = b"".join(struct.pack("<h", sample) for sample in samples)
    byte_rate = sample_rate * 2
    block_align = 2
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(pcm)),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16),
            b"data",
            struct.pack("<I", len(pcm)),
            pcm,
        ]
    )


def silent_wav_bytes(*, duration_seconds: float = 2.0) -> bytes:
    sample_rate = 8000
    samples = [0] * int(sample_rate * duration_seconds)
    pcm = b"".join(struct.pack("<h", sample) for sample in samples)
    byte_rate = sample_rate * 2
    block_align = 2
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(pcm)),
            b"WAVE",
            b"fmt ",
            struct.pack(
                "<IHHIIHH",
                16,
                1,
                1,
                sample_rate,
                byte_rate,
                block_align,
                16,
            ),
            b"data",
            struct.pack("<I", len(pcm)),
            pcm,
        ]
    )


def deterministic_embedding(text: str) -> list[float]:
    digest = hashlib.sha256(str(text or "").encode("utf-8")).digest()
    values: list[float] = []
    for index in range(0, 12, 4):
        raw = int.from_bytes(digest[index:index + 4], "big")
        values.append(round((raw % 1000) / 1000.0, 3))
    return values


def write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def write_text(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def write_json(path: Path, payload: object) -> Path:
    return write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")
