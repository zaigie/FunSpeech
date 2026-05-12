#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云实时语音识别 WebSocket 测试工具

用法:
    # 测试 WAV 文件
    python tests/test_aliyun_asr_ws.py --url ws://192.168.1.100:1247/ws/v1/asr --audio-file test.wav

    # 测试 PCM 文件 (需指定采样率和格式)
    python tests/test_aliyun_asr_ws.py --url ws://192.168.1.100:1247/ws/v1/asr --audio-file test.pcm --format pcm --sample-rate 16000

    # 交互式 — 从麦克风录音 (需要 sounddevice)
    python tests/test_aliyun_asr_ws.py --url ws://192.168.1.100:1247/ws/v1/asr --mic

依赖: pip install websockets soundfile sounddevice
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore
import websockets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _msg_id() -> str:
    return uuid.uuid4().hex[:32]


def _task_id() -> str:
    return uuid.uuid4().hex[:32]


async def transcribe_file(
    ws_url: str,
    audio_path: str,
    audio_format: str = "wav",
    sample_rate: int = 16000,
    chunk_ms: int = 120,
    token: str = "",
) -> None:
    """读取音频文件, 流式推送到 ASR WebSocket, 实时打印识别结果."""
    logger.info("读取音频: %s", audio_path)
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # 立体声→单声道
    logger.info("音频: %.1fs, %dHz, %d samples", len(audio) / sr, sr, len(audio))

    # 如果需要重采样 (简单线性插值, 仅用于测试)
    if sr != sample_rate:
        import scipy.signal as _signal

        audio = _signal.resample(audio, int(len(audio) * sample_rate / sr))
        sr = sample_rate
        logger.info("已重采样到 %dHz, %d samples", sample_rate, len(audio))

    # PCM int16
    audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    task_id = _task_id()
    chunk_samples = int(sr * chunk_ms / 1000)
    headers = {"X-NLS-Token": token} if token else {}

    async with websockets.connect(ws_url, extra_headers=headers) as ws:
        # 1) StartTranscription
        start = {
            "header": {
                "message_id": _msg_id(),
                "task_id": task_id,
                "namespace": "SpeechTranscriber",
                "name": "StartTranscription",
            },
            "payload": {
                "format": audio_format,
                "sample_rate": sample_rate,
                "enable_intermediate_result": True,
                "enable_punctuation_prediction": True,
                "enable_inverse_text_normalization": True,
                "max_sentence_silence": 800,
            },
        }
        await ws.send(json.dumps(start))
        logger.info("→ StartTranscription (task_id=%s)", task_id)

        # 2) 等待 TranscriptionStarted
        resp = json.loads(await ws.recv())
        hdr = resp.get("header", {})
        if hdr.get("name") == "TaskFailed":
            logger.error("❌ %s", hdr.get("status_text", resp))
            return
        logger.info("← %s", hdr.get("name", "?"))

        # 3) 发送音频 + 并发接收结果
        sentences: list[str] = []

        async def sender():
            """逐块发送音频数据"""
            for offset in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[offset : offset + chunk_samples].tobytes()
                await ws.send(chunk)
                await asyncio.sleep(chunk_ms / 1000 * 0.8)  # 模拟实时流
            # StopTranscription
            stop = {
                "header": {
                    "message_id": _msg_id(),
                    "task_id": task_id,
                    "namespace": "SpeechTranscriber",
                    "name": "StopTranscription",
                },
            }
            await ws.send(json.dumps(stop))
            logger.info("→ StopTranscription")

        async def receiver():
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    logger.debug("← 音频回执 %d bytes", len(msg))
                    continue
                data = json.loads(msg)
                hdr = data.get("header", {})
                name = hdr.get("name", "")
                pl = data.get("payload", {})

                if name == "SentenceBegin":
                    print(f"\n📝 句子 #{pl.get('index')} 开始")
                elif name == "TranscriptionResultChanged":
                    text = pl.get("result", "")
                    print(f"   ⏳ {text}", end="\r")
                elif name == "SentenceEnd":
                    text = pl.get("result", "")
                    sentences.append(text)
                    print(f"\r   ✅ {text}")
                elif name == "TranscriptionCompleted":
                    logger.info("← 识别完成")
                    break
                elif name == "TaskFailed":
                    logger.error("❌ %s", hdr.get("status_text", ""))
                    break

        await asyncio.gather(sender(), receiver())

        print(f"\n{'='*50}")
        print("全文:")
        print("".join(sentences))
        print(f"{'='*50}")


# ---------------------------------------------------------------------------
# 麦克风模式 (可选)
# ---------------------------------------------------------------------------


async def transcribe_mic(
    ws_url: str,
    sample_rate: int = 16000,
    device: int | None = None,
    token: str = "",
) -> None:
    """从麦克风实时录音 → WebSocket ASR"""
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        logger.error("麦克风模式需要 sounddevice: pip install sounddevice")
        return

    task_id = _task_id()
    block_samples = int(sample_rate * 0.12)  # 120ms 块
    headers = {"X-NLS-Token": token} if token else {}

    async with websockets.connect(ws_url, extra_headers=headers) as ws:
        start = {
            "header": {
                "message_id": _msg_id(),
                "task_id": task_id,
                "namespace": "SpeechTranscriber",
                "name": "StartTranscription",
            },
            "payload": {
                "format": "pcm",
                "sample_rate": sample_rate,
                "enable_intermediate_result": True,
                "enable_punctuation_prediction": True,
                "enable_inverse_text_normalization": True,
            },
        }
        await ws.send(json.dumps(start))
        resp = json.loads(await ws.recv())
        hdr = resp.get("header", {})
        if hdr.get("name") == "TaskFailed":
            logger.error("❌ %s", hdr.get("status_text", resp))
            return
        logger.info("← %s — 开始说话吧", hdr.get("name", "?"))

        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()
        mic_stopped = asyncio.Event()

        def audio_callback(indata, frames, _time, status):
            if status:
                logger.warning("mic: %s", status)
            chunk = indata[:, 0] if indata.ndim == 2 else indata.flatten()
            buf = (chunk * 32768).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(q.put_nowait, buf)

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_samples,
            callback=audio_callback,
            device=device,
        )

        async def sender():
            with stream:
                while not mic_stopped.is_set():
                    try:
                        buf = await asyncio.wait_for(q.get(), timeout=0.5)
                        await ws.send(buf)
                    except asyncio.TimeoutError:
                        continue
            stop = {
                "header": {
                    "message_id": _msg_id(),
                    "task_id": task_id,
                    "namespace": "SpeechTranscriber",
                    "name": "StopTranscription",
                },
            }
            await ws.send(json.dumps(stop))

        async def receiver():
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    continue
                data = json.loads(msg)
                hdr = data.get("header", {})
                name = hdr.get("name", "")
                pl = data.get("payload", {})

                if name == "SentenceBegin":
                    print(f"\n📝 #{pl.get('index')}")
                elif name == "TranscriptionResultChanged":
                    print(f"   ⏳ {pl.get('result', '')}", end="\r")
                elif name == "SentenceEnd":
                    print(f"\r   ✅ {pl.get('result', '')}")
                elif name == "TranscriptionCompleted":
                    break
                elif name == "TaskFailed":
                    logger.error("❌ %s", hdr.get("status_text", ""))
                    break

        # 按 Ctrl+C 停止录音
        send_task = asyncio.create_task(sender())
        try:
            await receiver()
        except KeyboardInterrupt:
            print("\n停止录音...")
        finally:
            mic_stopped.set()
            await send_task


# ---------------------------------------------------------------------------


async def main() -> None:
    ap = argparse.ArgumentParser(description="阿里云实时语音识别 WebSocket 测试")
    ap.add_argument("--url", default="ws://localhost:8000/ws/v1/asr", help="WebSocket URL")
    ap.add_argument("--audio-file", "-f", help="音频文件路径 (.wav / .pcm)")
    ap.add_argument("--format", default="wav", choices=["wav", "pcm"], help="音频格式")
    ap.add_argument("--sample-rate", type=int, default=16000, help="采样率")
    ap.add_argument("--chunk-ms", type=int, default=120, help="每块毫秒数")
    ap.add_argument("--token", default="", help="X-NLS-Token (可选)")
    ap.add_argument("--mic", action="store_true", help="麦克风实时录音模式")
    ap.add_argument("--mic-device", type=int, default=None, help="麦克风设备编号")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.mic:
        await transcribe_mic(
            args.url,
            sample_rate=args.sample_rate,
            device=args.mic_device,
            token=args.token,
        )
    elif args.audio_file:
        if not Path(args.audio_file).exists():
            logger.error("文件不存在: %s", args.audio_file)
            sys.exit(1)
        await transcribe_file(
            args.url,
            args.audio_file,
            audio_format=args.format,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            token=args.token,
        )
    else:
        ap.print_help()
        print("\n请指定 --audio-file 或 --mic")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))