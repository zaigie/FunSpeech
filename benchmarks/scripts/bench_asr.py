# -*- coding: utf-8 -*-
"""ASR 并发压力测试 — 离线 multipart 路径"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


AUDIO_DIR = Path(__file__).resolve().parent.parent / "audio"


async def one_request(client, url, headers, audio_path, req_id):
    t0 = time.perf_counter()
    try:
        with open(audio_path, "rb") as fp:
            files = {"audio": (audio_path.name, fp, "audio/wav")}
            data = {
                "enable_punctuation": "false",
                "enable_itn": "false",
                "enable_vad": "false",
                "sample_rate": "22050",
            }
            r = await client.post(url, files=files, data=data, headers=headers, timeout=180)
        elapsed = time.perf_counter() - t0
        ok = r.status_code == 200
        text = r.json().get("text", "") if ok else ""
        return {
            "req_id": req_id,
            "ok": ok,
            "status": r.status_code,
            "elapsed": elapsed,
            "text_len": len(text),
            "error": "" if ok else r.text[:200],
        }
    except Exception as exc:
        return {
            "req_id": req_id,
            "ok": False,
            "status": 0,
            "elapsed": time.perf_counter() - t0,
            "text_len": 0,
            "error": str(exc)[:200],
        }


async def run_concurrency(base_url, token, concurrency, audio_files):
    url = f"{base_url.rstrip('/')}/asr/file"
    headers = {"X-Internal-Token": token} if token else {}

    limits = httpx.Limits(
        max_connections=max(64, concurrency * 2),
        max_keepalive_connections=max(32, concurrency),
    )
    async with httpx.AsyncClient(limits=limits) as client:
        # 预热
        _ = await one_request(client, url, headers, audio_files[0], -1)

        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                one_request(
                    client,
                    url,
                    headers,
                    audio_files[i % len(audio_files)],
                    i,
                )
            )
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        total = time.perf_counter() - t0

    okay = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    def pct(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
        return xs[k]

    elapsed_list = [r["elapsed"] for r in okay]
    return {
        "concurrency": concurrency,
        "wall_sec": total,
        "ok": len(okay),
        "failed": len(failed),
        "req_per_sec": len(okay) / total if total > 0 else 0.0,
        "p50": pct(elapsed_list, 50),
        "p95": pct(elapsed_list, 95),
        "max": max(elapsed_list) if elapsed_list else 0.0,
        "min": min(elapsed_list) if elapsed_list else 0.0,
        "mean": statistics.mean(elapsed_list) if elapsed_list else 0.0,
        "errors": [r["error"] for r in failed][:3],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18001")
    ap.add_argument("--token", default="funspeech-internal")
    ap.add_argument("--concurrencies", default="1,2,4,8")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    audio_files = sorted(AUDIO_DIR.glob("sample_*.wav"))
    if not audio_files:
        raise SystemExit(
            f"no audio in {AUDIO_DIR}; run benchmarks/scripts/gen_audio.py first"
        )
    print(f"使用 {len(audio_files)} 段音频")

    cs = [int(x) for x in args.concurrencies.split(",")]
    runs = []
    for c in cs:
        print(f"--- 测试 concurrency={c} ---", flush=True)
        r = await run_concurrency(args.base_url, args.token, c, audio_files)
        runs.append(r)
        print(
            f"  wall={r['wall_sec']:.2f}s ok={r['ok']}/{r['ok'] + r['failed']} "
            f"req/s={r['req_per_sec']:.2f} p50={r['p50']:.2f}s p95={r['p95']:.2f}s "
            f"max={r['max']:.2f}s",
            flush=True,
        )
        if r["failed"]:
            print("  失败:", r["errors"])

    payload = json.dumps({"runs": runs}, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
