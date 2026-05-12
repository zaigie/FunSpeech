# -*- coding: utf-8 -*-
"""调 cosyvoice 子服务生成 N 段音频, 落盘到 benchmarks/audio/*.wav"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx


sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_tts import TEXTS  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent.parent / "audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def gen_one(client, base_url, token, idx, text):
    headers = {"X-Internal-Token": token}
    r = await client.post(
        f"{base_url}/tts/file",
        json={"text": text, "voice": "中文女", "speed": 1.0, "prompt": ""},
        headers=headers,
        timeout=180,
    )
    r.raise_for_status()
    p = OUT_DIR / f"sample_{idx:02d}.wav"
    p.write_bytes(r.content)
    return p, len(r.content)


async def main():
    base_url = os.environ.get("BASE_URL", "http://127.0.0.1:18004")
    token = os.environ.get("INTERNAL_SERVICE_TOKEN", "funspeech-internal")
    n = int(os.environ.get("N", "16"))

    limits = httpx.Limits(max_connections=8, max_keepalive_connections=4)
    async with httpx.AsyncClient(limits=limits) as client:
        # 串行生成, 不抢 GPU
        for i in range(n):
            p, sz = await gen_one(client, base_url, token, i, TEXTS[i % len(TEXTS)])
            print(f"  {p.name}  {sz / 1024:.1f}KB")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
