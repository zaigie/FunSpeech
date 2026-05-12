#!/usr/bin/env python3
"""测两个 cosyvoice 副本共享一张卡时的 RTF / 容量。

并发 N 个客户端, 轮流打到两个副本 URL, 统计单条耗时 + 估算 RTF。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time

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


async def one_request(client, url, headers, text, req_id):
    payload = {"text": text, "voice": "中文女", "speed": 1.0, "prompt": ""}
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, headers=headers, timeout=180)
        elapsed = time.perf_counter() - t0
        ok = r.status_code == 200
        native_sr = int(r.headers.get("X-Native-Sample-Rate", "24000"))
        audio_sec = max(0.0, (len(r.content) - 44) / 2 / native_sr) if ok else 0.0
        return {
            "req_id": req_id, "ok": ok, "status": r.status_code,
            "elapsed": elapsed, "audio_sec": audio_sec,
            "url": url,
        }
    except Exception as exc:
        return {
            "req_id": req_id, "ok": False, "status": 0,
            "elapsed": time.perf_counter() - t0, "audio_sec": 0.0,
            "url": url, "error": str(exc)[:200],
        }


async def run_concurrency(urls, token, concurrency):
    headers = {"X-Internal-Token": token} if token else {}
    limits = httpx.Limits(
        max_connections=max(64, concurrency * 2),
        max_keepalive_connections=max(32, concurrency),
    )
    async with httpx.AsyncClient(limits=limits) as client:
        # 预热: 每个副本各打一次
        for u in urls:
            await one_request(client, f"{u}/tts/file", headers, TEXTS[0], -1)

        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(
                one_request(
                    client,
                    f"{urls[i % len(urls)]}/tts/file",
                    headers,
                    TEXTS[i % len(TEXTS)],
                    i,
                )
            )
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    elapsed = [r["elapsed"] for r in ok]
    audio_sec = sum(r["audio_sec"] for r in ok)

    def pct(xs, p):
        if not xs: return 0.0
        xs = sorted(xs)
        return xs[max(0, min(len(xs)-1, int(round(p/100*(len(xs)-1)))))]

    # RTF per client (用户视角): mean elapsed / mean audio sec per request
    if ok:
        mean_audio = audio_sec / len(ok)
        mean_elapsed = statistics.mean(elapsed)
        rtf_per_client = mean_elapsed / mean_audio if mean_audio > 0 else 0
    else:
        rtf_per_client = 0
        mean_audio = 0
        mean_elapsed = 0

    return {
        "concurrency": concurrency,
        "wall_sec": wall,
        "ok": len(ok), "failed": len(results) - len(ok),
        "req_per_sec": len(ok) / wall if wall else 0,
        "audio_sec_total": audio_sec,
        "throughput_rtf": (audio_sec / wall) if wall else 0,  # 系统 RTF (越大越好)
        "rtf_per_client": rtf_per_client,  # 单客户端 RTF (≤1 = 实时)
        "mean_elapsed": mean_elapsed,
        "mean_audio_sec": mean_audio,
        "p50": pct(elapsed, 50),
        "p95": pct(elapsed, 95),
        "max": max(elapsed) if elapsed else 0,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="逗号分隔, 例如 http://localhost:18004,http://localhost:18014")
    ap.add_argument("--token", default="funspeech-internal")
    ap.add_argument("--concurrencies", default="1,2,4,8")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    urls = [u.strip().rstrip("/") for u in args.urls.split(",")]
    print(f"使用 {len(urls)} 个 URL: {urls}")

    cs = [int(x) for x in args.concurrencies.split(",")]
    runs = []
    for c in cs:
        print(f"--- N={c} ---", flush=True)
        r = await run_concurrency(urls, args.token, c)
        runs.append(r)
        print(
            f"  wall={r['wall_sec']:.2f}s ok={r['ok']}/{r['ok']+r['failed']} "
            f"sys_rtf={r['throughput_rtf']:.2f} "
            f"per_client_rtf={r['rtf_per_client']:.2f} "
            f"mean_lat={r['mean_elapsed']:.2f}s "
            f"p95={r['p95']:.2f}s"
        )

    payload = json.dumps({"urls": urls, "runs": runs}, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(payload)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"-> {args.out}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
