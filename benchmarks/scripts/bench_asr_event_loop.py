# -*- coding: utf-8 -*-
"""测试 ASR 子服务: 跑长音频推理期间, /health 是否还能正常响应"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx


AUDIO = Path(__file__).resolve().parent.parent / "audio" / "long_concat.wav"


async def asr_worker(client, base_url, token, idx):
    headers = {"X-Internal-Token": token} if token else {}
    t0 = time.perf_counter()
    with open(AUDIO, "rb") as fp:
        files = {"audio": (AUDIO.name, fp, "audio/wav")}
        data = {
            "enable_punctuation": "false",
            "enable_itn": "false",
            "enable_vad": "false",
            "sample_rate": "22050",
        }
        r = await client.post(
            f"{base_url}/asr/file",
            files=files,
            data=data,
            headers=headers,
            timeout=120,
        )
    return idx, time.perf_counter() - t0, r.status_code


async def health_probe(client, base_url, stop_evt, samples):
    while not stop_evt.is_set():
        t0 = time.perf_counter()
        try:
            r = await client.get(f"{base_url}/health", timeout=5)
            samples.append(((time.perf_counter() - t0) * 1000, r.status_code))
        except Exception:
            samples.append((9999.0, 0))
        await asyncio.sleep(0.02)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18001")
    ap.add_argument("--token", default="funspeech-internal")
    ap.add_argument("--n-asr", type=int, default=4)
    args = ap.parse_args()

    limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
    async with httpx.AsyncClient(limits=limits) as client:
        # 预热
        async with client.stream("GET", f"{args.base_url}/health", timeout=5) as _:
            pass

        stop_evt = asyncio.Event()
        samples = []
        probe = asyncio.create_task(
            health_probe(client, args.base_url, stop_evt, samples)
        )
        await asyncio.sleep(0.2)
        before = list(samples)
        samples.clear()

        tasks = [
            asyncio.create_task(asr_worker(client, args.base_url, args.token, i))
            for i in range(args.n_asr)
        ]
        results = await asyncio.gather(*tasks)

        await asyncio.sleep(0.3)
        stop_evt.set()
        await probe

    during = [s[0] for s in samples if s[1] == 200]

    def stats(xs, label):
        if not xs:
            print(f"  {label}: no samples")
            return
        xs.sort()
        p50 = xs[len(xs) // 2]
        p95 = xs[min(len(xs) - 1, int(len(xs) * 0.95))]
        print(
            f"  {label}: n={len(xs)} mean={statistics.mean(xs):.1f}ms "
            f"p50={p50:.1f}ms p95={p95:.1f}ms max={xs[-1]:.1f}ms"
        )

    print(f"ASR 期间 /health 延迟 (n-asr={args.n_asr}):")
    stats([s[0] for s in before if s[1] == 200], "ASR 开始前 (idle)")
    stats(during, "ASR 进行中       ")
    print("ASR 单条耗时:")
    for r in results:
        print(f"  idx={r[0]}  elapsed={r[1]:.2f}s  status={r[2]}")


if __name__ == "__main__":
    asyncio.run(main())
