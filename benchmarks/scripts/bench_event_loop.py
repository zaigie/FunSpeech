# -*- coding: utf-8 -*-
"""测试 event loop 是否被 GPU 推理阻塞

并发开 2 个 TTS 请求, 同时每 50ms 打一次 /health。如果 /health 延迟稳定 < 20ms,
说明 event loop 没被阻塞 (=patched 修对了)。如果 /health 延迟 跟着 TTS 一起涨到秒级,
说明 event loop 被 sync handler 阻塞了 (=baseline 的问题)。
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def tts_worker(client, base_url, token, idx, text):
    headers = {"X-Internal-Token": token} if token else {}
    t0 = time.perf_counter()
    r = await client.post(
        f"{base_url}/tts/file",
        json={"text": text, "voice": "中文女", "speed": 1.0, "prompt": ""},
        headers=headers,
        timeout=180,
    )
    return idx, time.perf_counter() - t0, r.status_code


async def health_probe(client, base_url, stop_evt, samples):
    while not stop_evt.is_set():
        t0 = time.perf_counter()
        try:
            r = await client.get(f"{base_url}/health", timeout=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            samples.append((elapsed_ms, r.status_code))
        except Exception:
            samples.append((9999.0, 0))
        await asyncio.sleep(0.05)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18004")
    ap.add_argument("--token", default="funspeech-internal")
    ap.add_argument("--n-tts", type=int, default=2)
    args = ap.parse_args()

    limits = httpx.Limits(max_connections=64, max_keepalive_connections=32)
    async with httpx.AsyncClient(limits=limits) as client:
        # 预热
        await client.post(
            f"{args.base_url}/tts/file",
            json={"text": "预热", "voice": "中文女", "speed": 1.0},
            headers={"X-Internal-Token": args.token},
            timeout=60,
        )

        stop_evt = asyncio.Event()
        samples: list = []
        probe = asyncio.create_task(
            health_probe(client, args.base_url, stop_evt, samples)
        )

        # 让 probe 稳定打一会
        await asyncio.sleep(0.5)
        baseline_samples = list(samples)
        samples.clear()

        # 并发开 TTS
        long_text = "人工智能技术正在快速改变各行各业的生产方式和服务模式。" * 2
        tts_tasks = [
            asyncio.create_task(
                tts_worker(client, args.base_url, args.token, i, long_text)
            )
            for i in range(args.n_tts)
        ]
        tts_results = await asyncio.gather(*tts_tasks)

        await asyncio.sleep(0.3)
        stop_evt.set()
        await probe

    during = [s[0] for s in samples if s[1] == 200]
    before = [s[0] for s in baseline_samples if s[1] == 200]

    def stats(xs, label):
        if not xs:
            print(f"  {label}: no samples")
            return
        xs.sort()
        p50 = xs[len(xs) // 2]
        p95 = xs[min(len(xs) - 1, int(len(xs) * 0.95))]
        max_ = xs[-1]
        mean = statistics.mean(xs)
        print(
            f"  {label}: n={len(xs)} mean={mean:.1f}ms "
            f"p50={p50:.1f}ms p95={p95:.1f}ms max={max_:.1f}ms"
        )

    print(f"TTS 期间 /health 延迟 (n-tts={args.n_tts}):")
    stats(before, "TTS 开始前 (idle)")
    stats(during, "TTS 进行中     ")
    print("TTS 单条耗时:")
    for r in tts_results:
        print(f"  idx={r[0]}  elapsed={r[1]:.2f}s  status={r[2]}")


if __name__ == "__main__":
    asyncio.run(main())
