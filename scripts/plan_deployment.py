#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署规划助手 — 零外部依赖 (仅 Python 3.8+ stdlib)

根据你的:
  - GPU 资源 (卡数、单卡显存)
  - 要启用的子服务 (funasr / dolphin / qwen3-asr / cosyvoice)
  - 每个服务的目标并发数 (同时活跃的客户端请求数, 不是 QPS)

输出:
  - 每个服务建议起几个副本
  - 副本怎么绑到宿主机 GPU (deploy.resources.reservations.devices.device_ids)
  - 当前目录下的完整 docker-compose.generated.yml
  - 估算的总显存占用

数据来源: 在 NVIDIA 4090 24G 上的实测 benchmarks (见 benchmarks/results/)。
其它显存档位用线性外推, 仅供参考, 真实部署后用 `nvidia-smi` 核对。

用法:
    python3 scripts/plan_deployment.py            # 交互式
    python3 scripts/plan_deployment.py --preset 8gb-single    # 用预设
    python3 scripts/plan_deployment.py --json input.json      # 从文件读输入
    python3 scripts/plan_deployment.py --help     # 看所有参数

生成文件若已存在, 会自动写成 docker-compose.generated.1.yml、
docker-compose.generated.2.yml 等, 不覆盖已有部署文件。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 实测基线 (4090 24G, idx=7)
# 数据出处: benchmarks/results/asr_funasr/, asr_qwen3/, tts/
# 想看原始数据见 benchmarks/README.md
# =============================================================================

# 单副本单卡的稳定吞吐 (req/s) — 实测值, 表示在该负载下系统不退化
THROUGHPUT_PER_REPLICA = {
    "funasr":    12.0,   # 实测 asr_patched_sem1 N=32: 12.6 req/s
    "dolphin":   12.0,   # 没单独测, 取与 funasr 同量级
    "qwen3-asr":  5.0,   # 实测 qwen3_patched_high N=64: 5.26 req/s
    "cosyvoice":  0.34,  # 实测 tts_patched_sem2 N=8: 0.34 req/s
}

# 单条请求平均延迟 (秒) — 实测值, 用来从"并发数"反推副本数
LATENCY_SEC_PER_REPLICA = {
    "funasr":     0.08,  # 实测 ~80 ms
    "dolphin":    0.08,  # 同量级
    "qwen3-asr":  0.20,  # 实测 ~190 ms (含 vLLM continuous batching)
    "cosyvoice":  3.50,  # 实测 ~3.5 s (autoregressive TTS, 慢)
}

# 单副本能"同时"处理的活跃请求数 (= GPU_INFERENCE_CONCURRENCY 或 vLLM batch)
# 这是 services/*/server.py 里硬编码的并发数, 见 deployment.md §4.2
PARALLEL_PER_REPLICA = {
    "funasr":     1,      # sem=1, GPU 串行
    "dolphin":    1,      # sem=1
    "qwen3-asr": 64,      # vLLM continuous batching, 实测 N=64 仍稳
    "cosyvoice":  2,      # sem=2, 实测最佳点 (sem=8 反而慢)
}

# cosyvoice (TTS) 的"实时容量" — 单副本能同时支持多少路 RTF ≤ 1 的客户端。
# RTF (Real-Time Factor) = 推理耗时 / 生成音频时长。RTF ≤ 1 表示推理 ≥ 实时,
# 客户端不会等待; RTF > 1 表示推理慢于实时, 流式播放会卡顿。
#
# 这是 TTS 用户最直观的指标 (用户问"我想同时跑 N 路 TTS 不卡, 要多少机器")。
#
# 实测来源: benchmarks/results/tts/tts_patched_sem2.json (N×并发, sem=2)
#   N=1: 单客户端 RTF ≈ 0.69 (3.48s 推 5s 音频, 性能过剩)
#   N=2: 单客户端 RTF ≈ 1.05 (5.28s 推 5s 音频, 刚好实时) ← 甜蜜点
#   N=4: 单客户端 RTF ≈ 1.75 (开始延迟于实时)
#   N=8: 单客户端 RTF ≈ 3.0  (严重落后)
# 取 RTF ≤ 1.1 (10% 容忍) 的最大 N 作为"实时容量"
TTS_REALTIME_CAPACITY_PER_REPLICA = 2  # cosyvoice 单副本 sem=2 时

# 单副本的显存 (GiB), 不含 vLLM KV cache 池
# 注意 qwen3-asr 的总显存 = WEIGHTS + KV_pool, KV_pool 见下面 qwen3_mem_util_for_card
MODEL_MEM_GIB = {
    # 模式: GiB
    "funasr": {"all": 3.0, "offline": 0.7, "realtime": 1.5},
    "dolphin": {"default": 0.6},
    "qwen3-asr": {"weights": 4.0},  # 只是权重, 不含 KV cache
    "cosyvoice": {"all": 5.0, "clone": 3.5, "sft": 1.5},
}

# 每个 CUDA context 在 GPU 上的额外开销 (经验值 ~300-500 MiB)
CUDA_CONTEXT_OVERHEAD_GIB = 0.4

DEFAULT_GENERATED_COMPOSE = "docker-compose.generated.yml"

SERVICE_TO_BASE_PORT = {
    "funasr": 8001,
    "dolphin": 8002,
    "qwen3-asr": 8003,
    "cosyvoice": 8004,
}

SERVICE_TO_SVC_NAME = {
    "funasr": "funasr",
    "dolphin": "dolphin",
    "qwen3-asr": "qwen3-asr",
    "cosyvoice": "cosyvoice",
}

SERVICE_TO_IMAGE = {
    "funasr": "docker.cnb.cool/nexa/funspeech/funasr:latest",
    "dolphin": "docker.cnb.cool/nexa/funspeech/dolphin:latest",
    "qwen3-asr": "docker.cnb.cool/nexa/funspeech/qwen3-asr:latest",
    "cosyvoice": "docker.cnb.cool/nexa/funspeech/cosyvoice:latest",
}

SERVICE_TO_BUILD_CONTEXT = {
    "funasr": "./services/funasr",
    "dolphin": "./services/dolphin",
    "qwen3-asr": "./services/qwen3_asr_vllm",
    "cosyvoice": "./services/cosyvoice",
}


# =============================================================================
# 数据结构
# =============================================================================


@dataclasses.dataclass
class GPU:
    idx: int            # 宿主机卡序号 (compose device_ids 用)
    total_gib: float    # 卡总显存
    label: str = ""     # 标签, 例如 "4090-24G"


@dataclasses.dataclass
class ServiceRequest:
    """服务需求。

    - ASR 服务 (funasr/dolphin/qwen3-asr): `concurrency` = 同时活跃的客户端请求数
      副本 = ceil(并发 / (并行容量 × 单位时间内处理批次)), 受 max_queue_sec 影响
    - TTS 服务 (cosyvoice): `concurrency` = 同时跑的实时 TTS 路数
      副本 = ceil(路数 / 实时容量), 实时容量 = TTS_REALTIME_CAPACITY_PER_REPLICA
      不依赖 max_queue_sec — 实时容量是工程实测点, 超了就 RTF>1 卡顿
    """

    name: str               # funasr / dolphin / qwen3-asr / cosyvoice
    concurrency: int        # ASR: 并发请求数; TTS: 实时路数
    mode: str               # all / offline / realtime / clone / sft / default

    @property
    def is_tts(self) -> bool:
        return self.name == "cosyvoice"

    @property
    def per_replica_qps(self) -> float:
        return THROUGHPUT_PER_REPLICA[self.name]

    @property
    def per_replica_parallel(self) -> int:
        return PARALLEL_PER_REPLICA[self.name]

    @property
    def per_replica_latency(self) -> float:
        return LATENCY_SEC_PER_REPLICA[self.name]

    @property
    def per_replica_realtime_capacity(self) -> Optional[int]:
        """TTS 单副本实时容量 (路数, RTF ≤ ~1.1)。ASR 返回 None。"""
        if self.is_tts:
            return TTS_REALTIME_CAPACITY_PER_REPLICA
        return None

    @property
    def replica_weight_gib(self) -> float:
        """单副本权重显存 (不含 qwen3-asr 的 KV pool)"""
        mem = MODEL_MEM_GIB[self.name]
        if self.name == "qwen3-asr":
            return mem["weights"]
        if self.name == "dolphin":
            return mem["default"]
        return mem.get(self.mode, mem.get("all", 1.0))

    def replicas_needed(self, max_queue_sec: float = 1.0) -> int:
        """副本数推算。

        TTS (cosyvoice):
            副本 = ceil(实时路数 / 实时容量)
            实时容量是单副本能保持 RTF ≤ 1.1 的客户端数 (实测 sem=2 时是 2)。
            想要 10 路实时 TTS → 5 副本。

        ASR (funasr/dolphin/qwen3-asr):
            副本 = ceil(并发 / (并行容量 × 单位时间内能处理的批次数))
            "批次数" = floor(max_queue_sec / 单条延迟), 至少 1。
            含义: 用户能接受最多排队 max_queue_sec 秒。
              - dolphin (单条 80ms, 容忍 1s): 批次 12 → 1 副本扛 12 并发
              - qwen3-asr (200ms, 1s): 批次 5 → 1 副本扛 64×5 = 320 并发
        """
        if self.concurrency <= 0:
            return 0

        if self.is_tts:
            cap = self.per_replica_realtime_capacity or 1
            return max(1, math.ceil(self.concurrency / cap))

        # ASR
        batches = max(1, int(max_queue_sec / self.per_replica_latency))
        capacity_per_replica = self.per_replica_parallel * batches
        return max(1, math.ceil(self.concurrency / capacity_per_replica))


@dataclasses.dataclass
class ReplicaPlacement:
    service: str        # 服务名
    replica_idx: int    # 0,1,...
    gpu_idx: int        # 卡序号
    mem_gib: float      # 这个副本在该卡上估算占用 (含 KV pool / CUDA context)
    extra: Dict         # 服务特定的额外信息, 例如 qwen3-asr 的 gpu_memory_utilization


# =============================================================================
# 规划逻辑
# =============================================================================


def qwen3_mem_util_for_card(card_gib: float, weights_gib: float = 4.0,
                            cohabit: bool = False) -> Tuple[float, float]:
    """根据卡总显存 + 是否与其它服务共置, 决定 QWEN3_ASR_GPU_MEM。

    `QWEN3_ASR_GPU_MEM` 是 vLLM 的 gpu_memory_utilization, 表示
    vLLM 在启动时**直接预留**这么大比例的卡显存做 KV cache 池
    (权重显存也算在这个比例里)。

    返回 (gpu_memory_utilization, 估算总占用 GiB)
    """
    if not cohabit:
        # 独占整卡, 越大越好 (跑 vLLM 价值最大), 但留点系统余量
        util = 0.85
    else:
        # 共置: 给 qwen3-asr 留 weights + 一点 KV 余量, 大约 6 GiB
        # util = 6 / card_gib
        reserve_gib = weights_gib + 2.0  # 4G 模型 + 2G KV
        util = round(reserve_gib / card_gib, 2)
        util = max(0.2, min(0.5, util))  # 共置场景上限 0.5
    actual_use = round(util * card_gib, 1)
    return util, actual_use


def estimate_replica_gib(svc: ServiceRequest, card_gib: float,
                        cohabit_count: int) -> Tuple[float, Dict]:
    """估算一个副本在某张卡上的总显存 (含 CUDA context 和服务特定开销)。

    cohabit_count: 本卡上同时跑了几个其它**服务**的副本 (不含自己)
    """
    extra: Dict = {}
    weights = svc.replica_weight_gib
    if svc.name == "qwen3-asr":
        util, total = qwen3_mem_util_for_card(
            card_gib, weights, cohabit=cohabit_count > 0
        )
        extra["QWEN3_ASR_GPU_MEM"] = util
        # vLLM 一开就直接占 total, 不只是 weights
        return total + CUDA_CONTEXT_OVERHEAD_GIB, extra
    return weights + CUDA_CONTEXT_OVERHEAD_GIB, extra


def plan(gpus: List[GPU], services: List[ServiceRequest],
         max_queue_sec: float = 1.0) -> Tuple[
    List[ReplicaPlacement], List[str]
]:
    """核心规划: 优先不同卡, 显存够时允许同卡多副本。

    实测 (benchmarks/results/tts_cohabit/, 4090 24G):
      - cosyvoice 同卡 2 副本: 系统总吞吐 = 单副本 ×1.7, per-client RTF 略升
        (单副本 N=2: RTF=1.10, 双副本 N=4: RTF=1.15 — 仍接近实时)
      - cosyvoice 同卡 3 副本: 系统总吞吐 = 单副本 ×2.3
      - 即每增加 1 个同卡副本, 实际容量 +0.8-0.9 副本 (经验值)

    排序策略:
      1. 优先放到没有同服务副本的"空"卡 (按剩余显存降序, worst-fit)
      2. 否则塞到已有同服务副本的卡 (按剩余显存降序)
      3. qwen3-asr 例外: vLLM KV pool 一启动就预占大半卡, 同卡再塞一个 vLLM
         会爆 (两个 vLLM 都尝试 0.85 显存 → OOM), 仍坚持不同卡

    返回 (placements, warnings)。
    """
    warnings: List[str] = []

    # 把每个服务展开成 N 个副本
    replicas: List[Tuple[ServiceRequest, int]] = []
    for svc in services:
        n = svc.replicas_needed(max_queue_sec=max_queue_sec)
        for i in range(n):
            replicas.append((svc, i))

    # 跟踪每张卡的已用 GiB 和已绑的服务集合
    gpu_used = {g.idx: 0.0 for g in gpus}
    gpu_services = {g.idx: set() for g in gpus}

    # 按"权重显存"降序排, 大的先放 (qwen3-asr 优先, 它一旦绑卡就吃掉大半)
    replicas.sort(key=lambda x: -x[0].replica_weight_gib)

    cohabit_warned_for = set()  # 已警告过的服务, 避免刷屏

    placements: List[ReplicaPlacement] = []
    for svc, repl_idx in replicas:
        # 候选卡: 装得下 + (非 qwen3-asr) 允许同卡多副本
        # 评分: 优先选没放过本服务的卡 (rank=0), 否则用同卡兜底 (rank=1)
        candidates = []
        for g in gpus:
            already_has_same_svc = svc.name in gpu_services[g.idx]
            # qwen3-asr 同卡再塞一个 vLLM 会爆显存, 强制不同卡
            if already_has_same_svc and svc.name == "qwen3-asr":
                continue
            cohabit_count = len(gpu_services[g.idx])
            need, extra = estimate_replica_gib(svc, g.total_gib, cohabit_count)
            avail = g.total_gib - gpu_used[g.idx]
            if avail >= need:
                rank = 1 if already_has_same_svc else 0  # 0 优先
                # 加 g.idx 给 tuple 一个明确的可比较的字段, 避免 tuple sort 时
                # 偶然回退到比较 GPU dataclass 报 TypeError
                candidates.append((rank, -(avail - need), g.idx, g, need, extra))

        if not candidates:
            same_svc_already = sum(
                1 for g in gpus if svc.name in gpu_services[g.idx]
            )
            if svc.is_tts:
                need_desc = f"{svc.concurrency} 路实时 TTS (RTF≤1)"
            else:
                need_desc = f"{svc.concurrency} 并发"
            if svc.name == "qwen3-asr" and same_svc_already > 0:
                warnings.append(
                    f"⚠️  {svc.name} 副本 #{repl_idx} 装不下: "
                    f"qwen3-asr 一卡只能放一个 (vLLM KV pool 占满整卡)。"
                    f" 想撑 {need_desc}, 需要再加 GPU。"
                )
            else:
                # 检查是不是被 qwen3-asr 的 KV pool 挤掉了
                has_qwen3 = any(
                    "qwen3-asr" in gpu_services[g.idx] for g in gpus
                )
                hint = ""
                if has_qwen3 and svc.name != "qwen3-asr":
                    hint = (
                        " (qwen3-asr 的 KV pool 占了卡的大部分显存; "
                        "可手动改 QWEN3_ASR_GPU_MEM 从 0.85 降到 0.5, "
                        "省出 ~9 GiB 给其它服务, 代价是 qwen3-asr 批处理能力下降)"
                    )
                warnings.append(
                    f"⚠️  {svc.name} 副本 #{repl_idx} 装不下: "
                    f"所有卡的剩余显存都 < {svc.replica_weight_gib:.1f} GiB"
                    f"{hint}"
                )
            continue

        # 排序: rank 升序 (优先 0=空卡), 然后 -剩余升序 (= 剩余降序, worst-fit)
        candidates.sort()
        rank, _, _, gpu, need, extra = candidates[0]
        # 同卡多副本时给一次提示
        if rank == 1 and svc.name not in cohabit_warned_for:
            warnings.append(
                f"ℹ️  {svc.name} 出现同卡多副本 (卡 {gpu.idx}): "
                f"系统总容量 ≈ 副本数 × 0.85, 单客户端 RTF 会略升 (实测 ~+5%); "
                f"如有空闲卡是不会触发的, 这是兜底"
            )
            cohabit_warned_for.add(svc.name)
        gpu_used[gpu.idx] += need
        gpu_services[gpu.idx].add(svc.name)
        placements.append(ReplicaPlacement(
            service=svc.name, replica_idx=repl_idx,
            gpu_idx=gpu.idx, mem_gib=need, extra=extra,
        ))

    return placements, warnings


# =============================================================================
# 渲染输出
# =============================================================================


def render(gpus: List[GPU], services: List[ServiceRequest],
           placements: List[ReplicaPlacement],
           warnings: List[str],
           max_queue_sec: float = 1.0) -> str:
    out = []
    out.append("=" * 70)
    out.append("FunSpeech 部署规划")
    out.append("=" * 70)

    out.append("\n## 输入")
    out.append(f"\nGPU 资源 ({len(gpus)} 张):")
    for g in gpus:
        label = f" [{g.label}]" if g.label else ""
        out.append(f"  - 卡 {g.idx}: {g.total_gib} GiB{label}")
    out.append(
        f"\n服务需求 (ASR 按并发数, TTS 按实时 RTF≤1 路数; 排队容忍 {max_queue_sec:.1f}s):"
    )
    for s in services:
        n = s.replicas_needed(max_queue_sec)
        if s.is_tts:
            rt_cap = s.per_replica_realtime_capacity or 1
            out.append(
                f"  - {s.name:<10s}  实时路数 {s.concurrency:>3d}, "
                f"单副本可保 {rt_cap} 路实时 (RTF≤1.1) → 需 {n} 副本 (mode={s.mode})"
            )
            if n > 0:
                total_cap = n * rt_cap
                out.append(
                    f"      估算: {n} 副本最多保 {total_cap} 路实时, "
                    f"实际跑 {s.concurrency} 路 → "
                    f"{'刚好实时' if total_cap == s.concurrency else '富余' if total_cap > s.concurrency else 'RTF>1 会卡顿'}"
                )
        else:
            cap = s.per_replica_parallel
            lat = s.per_replica_latency
            out.append(
                f"  - {s.name:<10s}  并发 {s.concurrency:>3d}, "
                f"单副本并行 {cap} (单条 ~{lat:.2f}s) → 需 {n} 副本 (mode={s.mode})"
            )
            if n > 0:
                batches = max(1, int(max_queue_sec / lat))
                total_cap = n * cap * batches
                wall_t = max(1, math.ceil(s.concurrency / (n * cap))) * lat
                out.append(
                    f"      估算: {n} 副本 {max_queue_sec:.0f}s 内可吃 {total_cap} 并发, "
                    f"实际 {s.concurrency} 并发 → 一批 ~{wall_t:.2f}s, "
                    f"稳态吞吐 {n * s.per_replica_qps:.1f} req/s"
                )

    # 警告
    if warnings:
        out.append("\n## ⚠️  警告")
        for w in warnings:
            out.append("  " + w)

    # 副本分布表
    out.append("\n## 副本 → 卡 分布")
    if not placements:
        out.append("  (无可放置的副本)")
    else:
        out.append(f"  {'服务':<12s}{'副本':<8s}{'卡':<8s}{'显存(GiB)':<14s}{'备注'}")
        for p in placements:
            extra_str = ""
            if p.extra.get("QWEN3_ASR_GPU_MEM") is not None:
                extra_str = f"QWEN3_ASR_GPU_MEM={p.extra['QWEN3_ASR_GPU_MEM']}"
            out.append(
                f"  {p.service:<12s}#{p.replica_idx:<6d}{p.gpu_idx:<8d}"
                f"{p.mem_gib:<14.1f}{extra_str}"
            )

    # 每卡汇总
    out.append("\n## 每卡显存预算")
    out.append(f"  {'卡':<6s}{'总':<10s}{'估算占用':<12s}{'剩余':<10s}{'服务'}")
    per_gpu_used: Dict[int, float] = {g.idx: 0.0 for g in gpus}
    per_gpu_svcs: Dict[int, List[str]] = {g.idx: [] for g in gpus}
    for p in placements:
        per_gpu_used[p.gpu_idx] += p.mem_gib
        per_gpu_svcs[p.gpu_idx].append(f"{p.service}#{p.replica_idx}")
    for g in gpus:
        used = per_gpu_used[g.idx]
        free = g.total_gib - used
        svcs = ", ".join(per_gpu_svcs[g.idx]) or "(空闲)"
        out.append(
            f"  {g.idx:<6d}{g.total_gib:<10.1f}{used:<12.1f}"
            f"{free:<10.1f}{svcs}"
        )

    # docker-compose 文件
    out.append("\n## 配置文件")
    out.append(
        f"  运行脚本会在当前目录生成完整 Compose 文件: {DEFAULT_GENERATED_COMPOSE}"
    )

    # 启动命令
    out.append("\n## 启动")
    profiles = set()
    for s in services:
        if s.name == "funasr":
            profiles.add("funasr")
        if s.name == "dolphin":
            profiles.add("dolphin")
    profile_args = "".join(f" --profile {p}" for p in sorted(profiles))
    out.append(
        f"  docker compose -f {DEFAULT_GENERATED_COMPOSE}{profile_args} up -d\n"
        f"  # 起来后用 `docker compose ps` 看健康状态, vLLM 加载约 60-120 秒"
    )

    out.append("\n" + "=" * 70)
    return "\n".join(out)


def render_full_compose(gpus: List[GPU], services: List[ServiceRequest],
                        placements: List[ReplicaPlacement],
                        warnings: Optional[List[str]] = None) -> str:
    """生成可独立使用的完整 docker compose 文件内容。"""
    del gpus  # 文件内容只依赖已完成的放置结果。

    by_svc: Dict[str, List[ReplicaPlacement]] = {}
    for p in placements:
        by_svc.setdefault(p.service, []).append(p)
    for items in by_svc.values():
        items.sort(key=lambda p: p.replica_idx)

    service_modes = {s.name: s.mode for s in services}
    gateway_env = _render_gateway_env(by_svc, service_modes)

    lines = [
        "# FunSpeech generated compose",
        "# Generated by scripts/plan_deployment.py. Do not edit by hand unless",
        "# you intend to keep this file as your deployment-specific compose.",
    ]
    if warnings:
        lines.append("#")
        lines.append("# Planning warnings:")
        for warning in warnings:
            lines.append(f"# - {warning}")
    lines.extend([
        "",
        "x-subservice-env: &subservice-env",
        "  INTERNAL_SERVICE_TOKEN: ${INTERNAL_SERVICE_TOKEN:-funspeech-internal}",
        "  LOG_LEVEL: ${LOG_LEVEL:-INFO}",
        "",
        "x-build-args: &build-args",
        "  HTTP_PROXY: ${HTTP_PROXY:-}",
        "  HTTPS_PROXY: ${HTTPS_PROXY:-}",
        "  NO_PROXY: ${NO_PROXY:-localhost,127.0.0.1,*.local}",
        "",
        "x-modelscope-cache: &modelscope-cache",
        "  type: bind",
        "  source: ${MODELSCOPE_CACHE:-~/.cache/modelscope/hub/models}",
        "  target: /root/.cache/modelscope/hub",
        "",
        "services:",
    ])
    lines.extend(_render_gateway_service(gateway_env))

    for svc_name in ("funasr", "dolphin", "qwen3-asr", "cosyvoice"):
        for placement in by_svc.get(svc_name, []):
            lines.extend(
                _render_subservice(
                    svc_name=svc_name,
                    mode=service_modes.get(svc_name, "default"),
                    placement=placement,
                )
            )

    lines.extend([
        "",
        "networks:",
        "  default:",
        "    name: funspeech-net",
        "",
    ])
    return "\n".join(lines)


def _render_gateway_env(by_svc: Dict[str, List[ReplicaPlacement]],
                        service_modes: Dict[str, str]) -> List[str]:
    lines = []
    for svc_name in ("funasr", "dolphin", "qwen3-asr", "cosyvoice"):
        placements = by_svc.get(svc_name)
        if not placements:
            continue
        env_var = f"{svc_name.upper().replace('-', '_')}_SERVICE_URLS"
        urls = ",".join(
            f"http://{SERVICE_TO_SVC_NAME[svc_name]}-{p.replica_idx}:"
            f"{SERVICE_TO_BASE_PORT[svc_name]}"
            for p in placements
        )
        lines.append(f"{env_var}: {urls}")
    lines.extend([
        "INTERNAL_SERVICE_TOKEN: ${INTERNAL_SERVICE_TOKEN:-funspeech-internal}",
        "SERVICE_REQUEST_TIMEOUT: \"120\"",
        f"ASR_MODEL_MODE: {service_modes.get('funasr', 'all')}",
        f"TTS_MODEL_MODE: {service_modes.get('cosyvoice', 'all')}",
    ])
    return lines


def _render_gateway_service(gateway_env: List[str]) -> List[str]:
    lines = [
        "  gateway:",
        "    build:",
        "      context: .",
        "      dockerfile: Dockerfile",
        "      args: *build-args",
        "    image: docker.cnb.cool/nexa/funspeech/gateway:latest",
        "    container_name: funspeech-gateway",
        "    ports:",
        "      - \"${GATEWAY_PORT:-8000}:8000\"",
        "    environment:",
    ]
    lines.extend(f"      {line}" for line in gateway_env)
    lines.extend([
        "      DEBUG: \"false\"",
        "      LOG_LEVEL: ${LOG_LEVEL:-INFO}",
        "      WORKERS: \"1\"",
        "      ASR_ENABLE_REALTIME_PUNC: ${ASR_ENABLE_REALTIME_PUNC:-false}",
        "      AUTO_LOAD_CUSTOM_ASR_MODELS: ${AUTO_LOAD_CUSTOM_ASR_MODELS:-}",
        "      APPTOKEN: ${APPTOKEN:-}",
        "      APPKEY: ${APPKEY:-}",
        "    volumes:",
        "      - ./temp:/app/temp",
        "      - ./data:/app/data",
        "      - ./logs:/app/logs",
        "    restart: unless-stopped",
    ])
    return lines


def _render_subservice(svc_name: str, mode: str,
                       placement: ReplicaPlacement) -> List[str]:
    compose_name = f"{SERVICE_TO_SVC_NAME[svc_name]}-{placement.replica_idx}"
    lines = [
        "",
        f"  {compose_name}:",
        "    build:",
        f"      context: {SERVICE_TO_BUILD_CONTEXT[svc_name]}",
        "      dockerfile: Dockerfile",
        "      args: *build-args",
        f"    image: {SERVICE_TO_IMAGE[svc_name]}",
        f"    container_name: funspeech-{compose_name}",
        "    environment:",
        "      <<: *subservice-env",
    ]
    lines.extend(_render_subservice_env(svc_name, mode, placement))
    lines.extend(_render_subservice_volumes(svc_name))
    lines.extend(_render_healthcheck(svc_name))
    lines.extend(_render_gpu_deploy(placement.gpu_idx))
    lines.extend([
        "    ulimits:",
        "      nofile:",
        "        soft: 65535",
        "        hard: 65535",
        "    restart: unless-stopped",
    ])
    if svc_name in ("funasr", "dolphin"):
        lines.append(f"    profiles: [\"{svc_name}\"]")
    return lines


def _render_subservice_env(svc_name: str, mode: str,
                           placement: ReplicaPlacement) -> List[str]:
    if svc_name == "funasr":
        return [
            "      PORT: \"8001\"",
            f"      ASR_MODEL_MODE: \"{mode}\"",
            "      ASR_DEVICE: cuda:0",
            "      CUDA_VISIBLE_DEVICES: \"0\"",
        ]
    if svc_name == "dolphin":
        return [
            "      PORT: \"8002\"",
            "      DOLPHIN_DEVICE: cuda",
            "      CUDA_VISIBLE_DEVICES: \"0\"",
        ]
    if svc_name == "qwen3-asr":
        gpu_mem = placement.extra.get("QWEN3_ASR_GPU_MEM", 0.8)
        return [
            "      PORT: \"8003\"",
            "      QWEN3_ASR_MODEL_ID: ${QWEN3_ASR_MODEL_ID:-Qwen/Qwen3-ASR-1.7B}",
            f"      QWEN3_ASR_GPU_MEM: \"{gpu_mem}\"",
            "      HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-}",
            "      TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE:-}",
            "      CUDA_VISIBLE_DEVICES: \"0\"",
        ]
    if svc_name == "cosyvoice":
        return [
            "      PORT: \"8004\"",
            f"      TTS_MODEL_MODE: \"{mode}\"",
            "      TTS_DEVICE: cuda:0",
            "      CLONE_MODEL_VERSION: ${CLONE_MODEL_VERSION:-cosyvoice3}",
            "      COSYVOICE3_MODEL_ID: ${COSYVOICE3_MODEL_ID:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}",
            "      TTS_LOAD_TRT: ${TTS_LOAD_TRT:-false}",
            "      TTS_ENABLE_FP16: ${TTS_ENABLE_FP16:-false}",
            "      TTS_LOAD_VLLM: ${TTS_LOAD_VLLM:-false}",
            "      VOICES_DIR: /app/voices",
            "      CUDA_VISIBLE_DEVICES: \"0\"",
        ]
    raise ValueError(f"unknown service: {svc_name}")


def _render_subservice_volumes(svc_name: str) -> List[str]:
    lines = [
        "    volumes:",
        "      - *modelscope-cache",
    ]
    if svc_name == "cosyvoice":
        lines.append("      - ./voices:/app/voices")
    return lines


def _render_healthcheck(svc_name: str) -> List[str]:
    port = SERVICE_TO_BASE_PORT[svc_name]
    retries = 12 if svc_name in ("qwen3-asr", "cosyvoice") else 8
    start_period = "300s" if svc_name in ("qwen3-asr", "cosyvoice") else "180s"
    return [
        "    healthcheck:",
        f"      test: [\"CMD\", \"curl\", \"-fsS\", \"http://localhost:{port}/health\"]",
        "      interval: 15s",
        "      timeout: 5s",
        f"      retries: {retries}",
        f"      start_period: {start_period}",
    ]


def _render_gpu_deploy(gpu_idx: int) -> List[str]:
    return [
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        f"              device_ids: [\"{gpu_idx}\"]",
        "              capabilities: [gpu]",
    ]


def _next_available_path(output_dir: Path,
                         filename: str = DEFAULT_GENERATED_COMPOSE) -> Path:
    base = output_dir / filename
    if not base.exists():
        return base
    suffix = base.suffix
    stem = base.stem
    idx = 1
    while True:
        candidate = output_dir / f"{stem}.{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def write_generated_compose(compose_text: str,
                            output_dir: Optional[Path] = None,
                            filename: str = DEFAULT_GENERATED_COMPOSE) -> Path:
    """写入当前目录, 若重名则使用 .1/.2 后缀避免覆盖。"""
    directory = Path.cwd() if output_dir is None else Path(output_dir)
    path = _next_available_path(directory, filename)
    path.write_text(compose_text, encoding="utf-8")
    return path


# =============================================================================
# 预设场景
# =============================================================================

PRESETS = {
    "8gb-single": {
        "gpus": [GPU(idx=0, total_gib=8, label="4060Ti-8G")],
        "services": [
            ServiceRequest("funasr", 20, "all"),
            ServiceRequest("cosyvoice", 2, "clone"),
        ],
        "desc": "8GB 单卡: funasr 20 并发 + cosyvoice 2 路实时 (qwen3-asr 装不下)",
    },
    "4090-single": {
        "gpus": [GPU(idx=0, total_gib=24, label="4090-24G")],
        "services": [
            ServiceRequest("qwen3-asr", 20, "default"),
            ServiceRequest("cosyvoice", 2, "clone"),
        ],
        "desc": "单张 4090 24G: qwen3-asr 20 并发 + cosyvoice 2 路实时 (qwen3-asr 抢占大半显存, cosyvoice 装不下)",
    },
    "4090-dual": {
        "gpus": [
            GPU(idx=0, total_gib=24, label="4090-24G"),
            GPU(idx=1, total_gib=24, label="4090-24G"),
        ],
        "services": [
            ServiceRequest("qwen3-asr", 40, "default"),
            ServiceRequest("cosyvoice", 4, "clone"),
        ],
        "desc": "双 4090: qwen3-asr 40 并发 + cosyvoice 4 路实时",
    },
    "4090-quad": {
        "gpus": [
            GPU(idx=i, total_gib=24, label="4090-24G") for i in range(4)
        ],
        "services": [
            ServiceRequest("funasr", 32, "all"),
            ServiceRequest("qwen3-asr", 64, "default"),
            ServiceRequest("cosyvoice", 4, "clone"),
        ],
        "desc": "4 张 4090: funasr 32 + qwen3-asr 64 + cosyvoice 4 并发",
    },
    "high-load": {
        "gpus": [
            GPU(idx=i, total_gib=24, label="4090-24G") for i in range(8)
        ],
        "services": [
            ServiceRequest("qwen3-asr", 128, "default"),
            ServiceRequest("cosyvoice", 8, "clone"),
        ],
        "desc": "8 张 4090: qwen3-asr 128 + cosyvoice 8 并发",
    },
}


# =============================================================================
# 交互式输入
# =============================================================================


def _ask(prompt: str, default: Optional[str] = None,
         validator=None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            v = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            print("")
            sys.exit(0)
        if not v and default is not None:
            v = default
        if not v:
            print("  (不能为空)")
            continue
        if validator:
            try:
                validator(v)
            except Exception as e:
                print(f"  无效: {e}")
                continue
        return v


def _ask_float(prompt: str, default: Optional[float] = None,
               min_val: float = 0) -> float:
    def _val(s):
        f = float(s)
        if f < min_val:
            raise ValueError(f">= {min_val}")
        return f
    s = _ask(prompt, str(default) if default is not None else None, _val)
    return float(s)


def _ask_int(prompt: str, default: Optional[int] = None,
             min_val: int = 0) -> int:
    return int(_ask_float(prompt, default, min_val))


def _ask_yn(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        v = input(f"{prompt} [{d}]: ").strip().lower()
        if not v:
            return default
        if v in ("y", "yes"):
            return True
        if v in ("n", "no"):
            return False


def interactive() -> Tuple[List[GPU], List[ServiceRequest]]:
    print("=" * 70)
    print("FunSpeech 部署规划助手")
    print("=" * 70)
    print()

    # GPU
    print("--- 步骤 1/2: 你有哪些 GPU? ---")
    n = _ask_int("GPU 数量", default=1, min_val=1)
    gpus = []
    same_mem = None
    if n > 1:
        if _ask_yn("所有卡显存一样吗?", default=True):
            same_mem = _ask_float("每张卡显存 (GiB)", default=24, min_val=1)
    for i in range(n):
        mem = same_mem if same_mem is not None else _ask_float(
            f"卡 {i} 显存 (GiB)", default=24, min_val=1
        )
        gpus.append(GPU(idx=i, total_gib=mem))
    print()

    # 服务
    print("--- 步骤 2/2: 启用哪些子服务, 每个的并发数? ---")
    print("(并发 = 同时活跃的客户端请求数, 不是 QPS)")
    print("  例: 20 个用户同时打开网页等结果 = 并发 20, 跟单条耗时无关")
    print()
    services = []

    if _ask_yn("启用 funasr (中文 ASR, 单条 ~80ms)?", default=True):
        c = _ask_int("  目标并发数", default=20, min_val=0)
        all_modes = _ask_yn("  需要流式 ASR 吗? (否=offline only, 省显存)",
                           default=True)
        services.append(ServiceRequest(
            "funasr", c, "all" if all_modes else "offline"
        ))

    if _ask_yn("启用 qwen3-asr (多语种/带标点 ASR, 显存大, 单条 ~190ms)?",
               default=False):
        c = _ask_int("  目标并发数 (单副本 vLLM 能并行 ~64)", default=20, min_val=0)
        services.append(ServiceRequest("qwen3-asr", c, "default"))

    if _ask_yn("启用 dolphin (多语种 ASR, 轻量, 单条 ~80ms)?", default=False):
        c = _ask_int("  目标并发数", default=10, min_val=0)
        services.append(ServiceRequest("dolphin", c, "default"))

    if _ask_yn("启用 cosyvoice (TTS)?", default=True):
        print("  TTS 按 RTF (推理耗时/音频时长) ≤ 1 的实时路数算容量")
        print("  实测: 单副本 (sem=2) 同时跑 2 路时 RTF≈1.05 (刚好实时),")
        print("        跑 4 路时 RTF≈1.75 (开始卡顿)")
        c = _ask_int(
            "  想同时支持多少路实时 TTS (单副本 = 2 路)",
            default=2, min_val=0,
        )
        clone = _ask_yn("  需要零样本克隆音色?", default=True)
        sft = _ask_yn("  需要预设音色 (中文女/男 等)?", default=True)
        if clone and sft:
            mode = "all"
        elif clone:
            mode = "clone"
        elif sft:
            mode = "sft"
        else:
            print("  你两个都不要, 那就不加 cosyvoice 了")
            mode = None
        if mode:
            services.append(ServiceRequest("cosyvoice", c, mode))

    if not services:
        print("没启用任何服务, 退出")
        sys.exit(0)

    print()
    return gpus, services


# =============================================================================
# JSON 输入 (CI / 脚本调用)
# =============================================================================


def from_json(path: str) -> Tuple[List[GPU], List[ServiceRequest]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gpus = [GPU(**g) for g in data["gpus"]]
    services = [ServiceRequest(**s) for s in data["services"]]
    return gpus, services


# =============================================================================
# main
# =============================================================================


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()),
                   help="使用内置预设场景")
    ap.add_argument("--json", help="从 JSON 文件读输入")
    ap.add_argument("--list-presets", action="store_true",
                   help="列出所有预设")
    ap.add_argument(
        "--max-queue-sec", type=float, default=1.0,
        help="用户能接受的排队时间 (秒, 默认 1.0)。短任务 (ASR) 这个值越大副本越少;"
             "长任务 (TTS 3.5s) 没影响, 副本数由并行容量决定。",
    )
    args = ap.parse_args()

    if args.list_presets:
        print("内置预设:")
        for k, v in PRESETS.items():
            print(f"  {k:<15s}  {v['desc']}")
        return

    if args.preset:
        p = PRESETS[args.preset]
        print(f"[预设: {args.preset}] {p['desc']}\n")
        gpus, services = p["gpus"], p["services"]
    elif args.json:
        gpus, services = from_json(args.json)
    else:
        gpus, services = interactive()

    placements, warnings = plan(gpus, services, max_queue_sec=args.max_queue_sec)
    compose_text = render_full_compose(gpus, services, placements, warnings)
    output_path = write_generated_compose(compose_text)

    profiles = set()
    for service in services:
        if service.name in ("funasr", "dolphin"):
            profiles.add(service.name)
    profile_args = "".join(f" --profile {p}" for p in sorted(profiles))

    print(f"已生成完整 Compose 文件: {output_path}")
    if warnings:
        print("\n规划警告:")
        for warning in warnings:
            print(f"  {warning}")
    print("\n静态检查:")
    print(f"  docker compose -f {output_path.name} config")
    print("\n启动:")
    print(f"  docker compose -f {output_path.name}{profile_args} up -d")


if __name__ == "__main__":
    main()
