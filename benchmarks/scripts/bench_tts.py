# -*- coding: utf-8 -*-
"""TTS 并发压力测试

打 N 个并发 HTTP POST /tts/file 到子服务, 统计:
  - 整体 throughput (req/s, 总音频 sec/wall sec)
  - 每请求耗时 p50 / p95 / max
  - 服务端实际并发处理是否成立 (并发耗时 vs 串行耗时)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import List

import httpx


TEXTS = [
    "人工智能技术正在快速改变各行各业的生产方式和服务模式。",
    "今天的天气非常好,适合出门散步,呼吸新鲜空气。",
    "请问从北京到上海的高铁需要多长时间,大概多少钱。",
    "深度学习模型的训练需要大量的计算资源和高质量的数据集。",
    "这家餐厅的招牌菜是红烧肉,味道鲜美,值得推荐给朋友。",
    "音乐能够帮助人们放松心情,缓解一天工作的疲劳和压力。",
    "晨跑是一种健康的生活习惯,有助于提高身体素质和精神状态。",
    "古代的诗人喜欢用山水来寄托自己的情感和人生感悟。",
    "新型电池技术的突破让电动汽车的续航里程大幅提升。",
    "周末和家人一起去公园野餐,是难得的休闲时光。",
    "学习一门新的编程语言能拓展自己的视野和职业发展空间。",
    "晚上九点之后不建议剧烈运动,可能影响睡眠质量。",
    "海边的日落非常壮观,金色的余晖洒在海面上分外美丽。",
    "图书馆是城市中难得的安静角落,适合阅读和思考。",
    "成功不会一蹴而就,需要持续的努力和长期的积累。",
    "保持微笑,生活会以同样的温暖回报每一个善良的人。",
]


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    text: str,
    voice: str,
    req_id: int,
) -> dict:
    payload = {"text": text, "voice": voice, "speed": 1.0, "prompt": ""}
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, headers=headers, timeout=180)
        elapsed = time.perf_counter() - t0
        ok = r.status_code == 200
        # 取出原生采样率, 算近似音频秒数
        native_sr = int(r.headers.get("X-Native-Sample-Rate", "24000"))
        audio_bytes = len(r.content) if ok else 0
        # WAV 头 + int16 mono: 估算时长
        approx_audio_sec = max(0.0, (audio_bytes - 44) / 2 / native_sr) if ok else 0.0
        return {
            "req_id": req_id,
            "ok": ok,
            "status": r.status_code,
            "elapsed": elapsed,
            "audio_sec": approx_audio_sec,
            "bytes": audio_bytes,
            "error": "" if ok else r.text[:200],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "req_id": req_id,
            "ok": False,
            "status": 0,
            "elapsed": elapsed,
            "audio_sec": 0.0,
            "bytes": 0,
            "error": str(exc)[:200],
        }


async def run_concurrency(
    base_url: str, token: str, voice: str, concurrency: int
) -> dict:
    url = f"{base_url.rstrip('/')}/tts/file"
    headers = {"X-Internal-Token": token} if token else {}

    # 复用 client + 高 limits, 才能真正并发出去
    limits = httpx.Limits(
        max_connections=max(64, concurrency * 2),
        max_keepalive_connections=max(32, concurrency),
    )
    async with httpx.AsyncClient(limits=limits) as client:
        # 预热: 第 1 个请求先单独跑, 让模型缓存 ready
        _ = await one_request(client, url, headers, TEXTS[0], voice, -1)

        # 真正并发批次
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                one_request(
                    client,
                    url,
                    headers,
                    TEXTS[i % len(TEXTS)],
                    voice,
                    i,
                )
            )
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        total = time.perf_counter() - t0

    okay = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    elapsed_list = [r["elapsed"] for r in okay]
    audio_sec = sum(r["audio_sec"] for r in okay)

    def pct(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
        return xs[k]

    return {
        "concurrency": concurrency,
        "wall_sec": total,
        "ok": len(okay),
        "failed": len(failed),
        "req_per_sec": len(okay) / total if total > 0 else 0.0,
        "audio_sec_total": audio_sec,
        "rtf": (audio_sec / total) if total > 0 else 0.0,  # > 1 表示比实时快
        "p50": pct(elapsed_list, 50),
        "p95": pct(elapsed_list, 95),
        "max": max(elapsed_list) if elapsed_list else 0.0,
        "min": min(elapsed_list) if elapsed_list else 0.0,
        "mean": statistics.mean(elapsed_list) if elapsed_list else 0.0,
        "errors": [r["error"] for r in failed][:3],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18004")
    ap.add_argument("--token", default="funspeech-internal")
    ap.add_argument("--voice", default="中文女")
    ap.add_argument(
        "--concurrencies", default="1,2,4,8", help="逗号分隔, 例如 1,2,4,8"
    )
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    cs = [int(x) for x in args.concurrencies.split(",")]
    results = []
    for c in cs:
        print(f"--- 测试 concurrency={c} ---", flush=True)
        r = await run_concurrency(args.base_url, args.token, args.voice, c)
        results.append(r)
        print(
            f"  wall={r['wall_sec']:.2f}s  ok={r['ok']}/{r['ok'] + r['failed']}  "
            f"req/s={r['req_per_sec']:.2f}  rtf={r['rtf']:.2f}  "
            f"p50={r['p50']:.2f}s  p95={r['p95']:.2f}s  max={r['max']:.2f}s",
            flush=True,
        )
        if r["failed"]:
            print("  失败样例:", r["errors"])

    payload = json.dumps({"runs": results}, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(payload)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
