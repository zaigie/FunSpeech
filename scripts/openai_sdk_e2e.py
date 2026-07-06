#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end checks for the OpenAI-compatible audio API using openai-python."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import httpx
from openai import OpenAI


DEFAULT_BASE_URL = "http://127.0.0.1:1247/v1"
DEFAULT_AUDIO = Path("benchmarks/audio/sample_00.wav")
DEFAULT_OUTPUT_DIR = Path("temp/openai_sdk_e2e")
DEFAULT_TTS_TEXT = "你好，这是 OpenAI SDK 流式语音合成测试。"


def _client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False, timeout=timeout),
    )


def _write_binary_response(response: Any, output_path: Path) -> int:
    total = 0
    with output_path.open("wb") as fp:
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            fp.write(chunk)
            total += len(chunk)
    if total <= 0:
        raise RuntimeError(f"empty binary response: {output_path}")
    return total


def _iter_sse_payloads(lines: Iterable[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads


def _event_value(event: Any, name: str) -> Any:
    if hasattr(event, name):
        return getattr(event, name)
    if hasattr(event, "model_dump"):
        return event.model_dump().get(name)
    if isinstance(event, dict):
        return event.get(name)
    return None


def run_tts_file(client: OpenAI, voice: str, text: str, output_dir: Path) -> None:
    output_path = output_dir / "tts_file.wav"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="wav",
    ) as response:
        written = _write_binary_response(response, output_path)
    print(f"[PASS] TTS file wav: {written} bytes -> {output_path}")


def run_tts_audio_stream(client: OpenAI, voice: str, text: str, output_dir: Path) -> None:
    output_path = output_dir / "tts_stream_audio.pcm"
    chunks = 0
    bytes_written = 0
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="pcm",
        stream_format="audio",
    ) as response:
        with output_path.open("wb") as fp:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                fp.write(chunk)
                chunks += 1
                bytes_written += len(chunk)
    if bytes_written <= 0:
        raise RuntimeError("empty TTS audio stream")
    print(
        f"[PASS] TTS stream_format=audio: {chunks} chunks, "
        f"{bytes_written} bytes -> {output_path}"
    )


def run_tts_sse_stream(client: OpenAI, voice: str, text: str, output_dir: Path) -> None:
    output_path = output_dir / "tts_stream_sse.jsonl"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="pcm",
        stream_format="sse",
    ) as response:
        payloads = _iter_sse_payloads(response.iter_lines())
    if not payloads:
        raise RuntimeError("empty TTS SSE stream")
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
        encoding="utf-8",
    )
    event_types = [str(item.get("type")) for item in payloads]
    if "speech.audio.done" not in event_types:
        raise RuntimeError(f"TTS SSE stream did not finish cleanly: {event_types}")
    print(f"[PASS] TTS stream_format=sse: {event_types} -> {output_path}")


def run_asr_json(client: OpenAI, audio_path: Path) -> str:
    with audio_path.open("rb") as fp:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=fp,
            response_format="json",
            language="zh",
        )
    text = getattr(result, "text", None)
    if not text and hasattr(result, "model_dump"):
        text = result.model_dump().get("text")
    if not text:
        raise RuntimeError(f"empty ASR result: {result!r}")
    print(f"[PASS] ASR json: {text}")
    return str(text)


def run_asr_stream(client: OpenAI, audio_path: Path, output_dir: Path) -> None:
    output_path = output_dir / "asr_stream_events.jsonl"
    events: list[dict[str, Any]] = []
    with audio_path.open("rb") as fp:
        stream = client.audio.transcriptions.create(
            model="whisper-1",
            file=fp,
            stream=True,
            language="zh",
        )
        for event in stream:
            item: dict[str, Any]
            if hasattr(event, "model_dump"):
                item = event.model_dump()
            elif isinstance(event, dict):
                item = dict(event)
            else:
                item = {"type": _event_value(event, "type"), "repr": repr(event)}
            events.append(item)

    if not events:
        raise RuntimeError("empty ASR stream")

    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in events) + "\n",
        encoding="utf-8",
    )
    event_types = [str(item.get("type")) for item in events]
    if "transcript.text.done" not in event_types:
        raise RuntimeError(f"ASR stream did not finish cleanly: {event_types}")
    text = next((str(item.get("text")) for item in events if item.get("text")), "")
    print(f"[PASS] ASR stream=true: {event_types}, text={text} -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI SDK base_url, including the /v1 prefix.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("APPTOKEN") or "dummy",
        help="Bearer token for the OpenAI-compatible gateway.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path(os.getenv("OPENAI_E2E_AUDIO", DEFAULT_AUDIO)),
        help="Audio file used for ASR checks.",
    )
    parser.add_argument(
        "--voice",
        default=os.getenv("OPENAI_TTS_VOICE", "中文女"),
        help="TTS voice id/name.",
    )
    parser.add_argument(
        "--text",
        default=os.getenv("OPENAI_TTS_TEXT", DEFAULT_TTS_TEXT),
        help="Text used for TTS checks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("OPENAI_E2E_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OPENAI_E2E_TIMEOUT", "300")),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--suite",
        choices=("all", "tts", "asr"),
        default=os.getenv("OPENAI_E2E_SUITE", "all"),
        help="Subset to run. Use tts/asr when only one GPU service should be up.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"base_url={args.base_url}")
    print(f"audio={args.audio}")
    print(f"output_dir={args.output_dir}")

    client = _client(args.base_url, args.api_key, args.timeout)
    if args.suite in ("all", "tts"):
        run_tts_file(client, args.voice, args.text, args.output_dir)
        run_tts_audio_stream(client, args.voice, args.text, args.output_dir)
        run_tts_sse_stream(client, args.voice, args.text, args.output_dir)
    if args.suite in ("all", "asr"):
        run_asr_json(client, args.audio)
        run_asr_stream(client, args.audio, args.output_dir)


if __name__ == "__main__":
    main()
