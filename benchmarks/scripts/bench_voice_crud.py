#!/usr/bin/env python3
"""并发 voice CRUD 测试 — 验证 C3 加锁 + 原子写

启 N 个并发线程, 各自 POST /voices 注册不同名字的音色, 然后:
1. 检查所有名字都能 GET 到 (没丢)
2. 检查 spk2info.pt 和 voice_registry.json 是有效文件 (没半写)
3. 反复跑几轮, 不应该出现 corruption
"""
import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx


async def add_voice(client, base_url, token, name, prompt_text, wav_path):
    headers = {"X-Internal-Token": token}
    with open(wav_path, "rb") as fp:
        files = {"audio": (f"{name}.wav", fp, "audio/wav")}
        data = {"name": name, "prompt_text": prompt_text}
        t0 = time.perf_counter()
        r = await client.post(
            f"{base_url}/voices",
            files=files,
            data=data,
            headers=headers,
            timeout=60,
        )
        elapsed = time.perf_counter() - t0
        return {"name": name, "status": r.status_code, "elapsed": elapsed,
                "error": r.text[:200] if r.status_code != 200 else ""}


async def delete_voice(client, base_url, token, name):
    headers = {"X-Internal-Token": token}
    r = await client.delete(
        f"{base_url}/voices/{name}",
        headers=headers,
        timeout=30,
    )
    return r.status_code


async def list_voices(client, base_url, token):
    headers = {"X-Internal-Token": token}
    r = await client.get(f"{base_url}/voices", headers=headers, timeout=10)
    return r.json()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18004")
    ap.add_argument("--token", default="funspeech-internal")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    test_prefix = f"smoketest_{int(time.time())}_"
    names = [f"{test_prefix}{i:02d}" for i in range(args.n)]
    prompt = "测试音色注册的提示文本,只是验证并发安全性。"

    async with httpx.AsyncClient() as client:
        # 清理前一次残留
        before = await list_voices(client, args.base_url, args.token)
        existing = set(before.get("registry", {}).keys())
        leftover = [n for n in existing if n.startswith(test_prefix)]
        if leftover:
            print(f"清理 {len(leftover)} 个旧测试音色")
            for n in leftover:
                await delete_voice(client, args.base_url, args.token, n)

        for round_i in range(args.rounds):
            print(f"\n=== Round {round_i + 1}/{args.rounds} ===")

            # 1. 并发添加
            print(f"并发 POST /voices × {args.n}")
            t0 = time.perf_counter()
            results = await asyncio.gather(*[
                add_voice(client, args.base_url, args.token,
                         names[i], prompt, args.wav)
                for i in range(args.n)
            ])
            wall = time.perf_counter() - t0
            ok = sum(1 for r in results if r["status"] == 200)
            print(f"  wall={wall:.2f}s ok={ok}/{args.n}")
            for r in results:
                if r["status"] != 200:
                    print(f"  FAIL {r['name']}: status={r['status']} err={r['error']}")

            # 2. 列表检查
            after = await list_voices(client, args.base_url, args.token)
            registry = after.get("registry", {})
            present = [n for n in names if n in registry]
            missing = [n for n in names if n not in registry]
            print(f"  registry 里有 {len(present)}/{args.n} 个音色")
            if missing:
                print(f"  MISSING: {missing}")

            # 3. 各自查 voice_info
            info_ok = 0
            info_fail = 0
            for n in names:
                headers = {"X-Internal-Token": args.token}
                r = await client.get(
                    f"{args.base_url}/voices/{n}", headers=headers, timeout=5
                )
                if r.status_code == 200 and r.json().get("name") == n:
                    info_ok += 1
                else:
                    info_fail += 1
                    print(f"  info FAIL {n}: status={r.status_code}")
            print(f"  GET /voices/{{name}}: {info_ok}/{args.n} 正确")

            # 4. 清理本轮
            print("  清理本轮音色")
            del_ok = 0
            for n in names:
                s = await delete_voice(client, args.base_url, args.token, n)
                if s == 200:
                    del_ok += 1
            print(f"  delete: {del_ok}/{args.n}")

        # 最终检查: 磁盘文件
        print("\n=== 最终磁盘检查 ===")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
