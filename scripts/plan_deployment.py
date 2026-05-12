#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署规划助手 — 零外部依赖 (仅 Python 3.8+ stdlib)

根据你的:
  - GPU 资源 (卡数、单卡显存)
  - 要启用的子服务 (funasr / dolphin / qwen3-asr / cosyvoice)
  - 每个服务的目标 QPS 或并发路数

输出:
  - 每个服务建议起几个副本
  - 副本怎么绑卡 (CUDA_VISIBLE_DEVICES 分布)
  - .env / docker-compose 配置片段
  - 估算的总显存占用

数据来源: 在 NVIDIA 4090 24G 上的实测 benchmarks (见 benchmarks/results/)。
其它显存档位用线性外推, 仅供参考, 真实部署后用 `nvidia-smi` 核对。

用法:
    python3 scripts/plan_deployment.py            # 交互式
    python3 scripts/plan_deployment.py --preset 8gb-single    # 用预设
    python3 scripts/plan_deployment.py --json input.json      # 从文件读输入
    python3 scripts/plan_deployment.py --help     # 看所有参数
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import textwrap
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


# =============================================================================
# 数据结构
# =============================================================================


@dataclasses.dataclass
class GPU:
    idx: int            # 卡序号 (CUDA_VISIBLE_DEVICES 用)
    total_gib: float    # 卡总显存
    label: str = ""     # 标签, 例如 "4090-24G"


@dataclasses.dataclass
class ServiceRequest:
    name: str           # funasr / dolphin / qwen3-asr / cosyvoice
    target_qps: float   # 期望 QPS (req/s)
    mode: str           # all / offline / realtime / clone / sft / default
    # cosyvoice 的 mode=all 包含两个模型 (sft + clone), 显存翻倍
    # funasr 的 mode=offline 只加载离线模型, 显存约 1/4

    @property
    def per_replica_qps(self) -> float:
        return THROUGHPUT_PER_REPLICA[self.name]

    @property
    def replica_weight_gib(self) -> float:
        """单副本权重显存 (不含 qwen3-asr 的 KV pool)"""
        mem = MODEL_MEM_GIB[self.name]
        if self.name == "qwen3-asr":
            return mem["weights"]
        if self.name == "dolphin":
            return mem["default"]
        return mem.get(self.mode, mem.get("all", 1.0))

    def replicas_needed(self) -> int:
        if self.target_qps <= 0:
            return 0
        return max(1, math.ceil(self.target_qps / self.per_replica_qps))


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


def plan(gpus: List[GPU], services: List[ServiceRequest]) -> Tuple[
    List[ReplicaPlacement], List[str]
]:
    """核心规划: first-fit decreasing 启发式。

    关键约束 (基于实测):
    1. 同一服务的多个副本**不能放同一张卡** — 否则它们抢 GPU, 总吞吐 ≈ 1 副本
       qwen3-asr 同卡多副本还会让 vLLM KV pool 各自变小, 效率更差
    2. 副本之间的"权重显存"是真实占用, vLLM 还要加 KV pool
    3. 同卡可以共置不同服务 (例如 funasr + cosyvoice), 累加显存

    返回 (placements, warnings)。
    """
    warnings: List[str] = []

    # 把每个服务展开成 N 个副本
    replicas: List[Tuple[ServiceRequest, int]] = []
    for svc in services:
        n = svc.replicas_needed()
        for i in range(n):
            replicas.append((svc, i))

    # 跟踪每张卡的已用 GiB 和已绑的服务集合
    gpu_used = {g.idx: 0.0 for g in gpus}
    gpu_services = {g.idx: set() for g in gpus}

    # 按"权重显存"降序排, 大的先放 (qwen3-asr 优先, 它一旦绑卡就吃掉大半)
    replicas.sort(key=lambda x: -x[0].replica_weight_gib)

    placements: List[ReplicaPlacement] = []
    for svc, repl_idx in replicas:
        # 候选卡: 必须没放过这个服务 + 装得下
        candidates = []
        for g in gpus:
            if svc.name in gpu_services[g.idx]:
                continue  # 同服务副本不同卡
            cohabit_count = len(gpu_services[g.idx])
            need, extra = estimate_replica_gib(svc, g.total_gib, cohabit_count)
            avail = g.total_gib - gpu_used[g.idx]
            if avail >= need:
                candidates.append((avail - need, g, need, extra))
        if not candidates:
            # 区分两种"放不下": 显存不够 vs 卡数不够
            same_svc_already = sum(
                1 for g in gpus if svc.name in gpu_services[g.idx]
            )
            if same_svc_already > 0:
                warnings.append(
                    f"⚠️  {svc.name} 副本 #{repl_idx} 装不下: "
                    f"已用完所有 {len(gpus)} 张卡 (每张最多 1 个 {svc.name} 副本)。"
                    f" 想达到目标 QPS, 需要再加 GPU。"
                )
            else:
                # 检查是不是被 qwen3-asr 的 KV pool 挤掉了
                has_qwen3_taking_all = any(
                    "qwen3-asr" in gpu_services[g.idx]
                    for g in gpus
                )
                hint = ""
                if has_qwen3_taking_all and svc.name != "qwen3-asr":
                    hint = (
                        " (qwen3-asr 的 KV pool 占了卡的大部分显存; "
                        "可手动改 QWEN3_ASR_GPU_MEM 从 0.85 降到 0.5, "
                        "省出 ~9 GiB 给其它服务, 代价是 qwen3-asr 批处理能力下降)"
                    )
                warnings.append(
                    f"⚠️  {svc.name} 副本 #{repl_idx} 装不下: "
                    f"没卡有 {svc.replica_weight_gib:.1f}+ GiB 空间放它"
                    f"{hint}"
                )
            continue
        # 取剩余最大的卡 (worst-fit, 让显存更均匀分布)
        candidates.sort(key=lambda x: -x[0])
        _, gpu, need, extra = candidates[0]
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
           warnings: List[str]) -> str:
    out = []
    out.append("=" * 70)
    out.append("FunSpeech 部署规划")
    out.append("=" * 70)

    out.append("\n## 输入")
    out.append(f"\nGPU 资源 ({len(gpus)} 张):")
    for g in gpus:
        label = f" [{g.label}]" if g.label else ""
        out.append(f"  - 卡 {g.idx}: {g.total_gib} GiB{label}")
    out.append("\n服务需求:")
    for s in services:
        per = THROUGHPUT_PER_REPLICA[s.name]
        n = s.replicas_needed()
        out.append(
            f"  - {s.name:<10s}  目标 {s.target_qps:>5.1f} req/s, "
            f"单副本 {per:>5.1f} req/s → 需 {n} 副本 "
            f"(mode={s.mode})"
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

    # docker-compose 片段
    out.append("\n## 配置片段")
    out.append(_render_compose_snippet(gpus, services, placements))

    # 启动命令
    out.append("\n## 启动")
    profiles = set()
    for s in services:
        if s.name == "dolphin":
            profiles.add("dolphin")
        if s.name == "qwen3-asr":
            profiles.add("qwen3-asr")
    profile_args = "".join(f" --profile {p}" for p in sorted(profiles))
    out.append(
        f"  docker compose{profile_args} up -d\n"
        f"  # 起来后用 `docker compose ps` 看健康状态, vLLM 加载约 60-120 秒"
    )

    out.append("\n" + "=" * 70)
    return "\n".join(out)


def _render_compose_snippet(gpus: List[GPU], services: List[ServiceRequest],
                            placements: List[ReplicaPlacement]) -> str:
    """生成 docker-compose.override.yml 和网关 env 片段"""
    # 按 (service, replica_idx) 索引
    by_svc: Dict[str, List[ReplicaPlacement]] = {}
    for p in placements:
        by_svc.setdefault(p.service, []).append(p)

    SERVICE_TO_BASE_PORT = {
        "funasr": 8001, "dolphin": 8002,
        "qwen3-asr": 8003, "cosyvoice": 8004,
    }
    SERVICE_TO_SVC_NAME = {
        "funasr": "funasr", "dolphin": "dolphin",
        "qwen3-asr": "qwen3-asr", "cosyvoice": "cosyvoice",
    }
    # docker-compose 里默认服务名带 `-0` 后缀, 多副本时是 -0, -1, ...
    lines = ["```yaml", "# === docker-compose.override.yml ===", "services:"]

    for svc_name in ("funasr", "dolphin", "qwen3-asr", "cosyvoice"):
        if svc_name not in by_svc:
            continue
        for p in by_svc[svc_name]:
            compose_name = f"{SERVICE_TO_SVC_NAME[svc_name]}-{p.replica_idx}"
            lines.append(f"  {compose_name}:")
            env_block = [f"      CUDA_VISIBLE_DEVICES: \"{p.gpu_idx}\""]
            if p.extra.get("QWEN3_ASR_GPU_MEM") is not None:
                env_block.append(
                    f"      QWEN3_ASR_GPU_MEM: \"{p.extra['QWEN3_ASR_GPU_MEM']}\""
                )
            # 服务 mode
            svc_req = next(s for s in services if s.name == svc_name)
            if svc_name == "funasr":
                env_block.append(f"      ASR_MODEL_MODE: \"{svc_req.mode}\"")
            elif svc_name == "cosyvoice":
                env_block.append(f"      TTS_MODEL_MODE: \"{svc_req.mode}\"")
            lines.append("    environment:")
            lines.extend(env_block)
            lines.append("    deploy:")
            lines.append("      resources:")
            lines.append("        reservations:")
            lines.append("          devices:")
            lines.append(
                "            - {driver: nvidia, "
                f"device_ids: [\"{p.gpu_idx}\"], capabilities: [gpu]}}"
            )
            # 多副本时容器名 / 端口要错开
            if p.replica_idx > 0:
                host_port = SERVICE_TO_BASE_PORT[svc_name] + p.replica_idx * 10
                lines.append(f"    container_name: funspeech-{compose_name}")
    lines.append("```")

    # 网关 env
    lines.append("\n```bash")
    lines.append("# === .env (网关) ===")
    for svc_name in ("funasr", "dolphin", "qwen3-asr", "cosyvoice"):
        if svc_name not in by_svc:
            continue
        env_var = f"{svc_name.upper().replace('-', '_')}_SERVICE_URLS"
        urls = ",".join(
            f"http://{SERVICE_TO_SVC_NAME[svc_name]}-{p.replica_idx}:"
            f"{SERVICE_TO_BASE_PORT[svc_name]}"
            for p in by_svc[svc_name]
        )
        lines.append(f"{env_var}={urls}")
    lines.append("```")

    return "\n".join(lines)


# =============================================================================
# 预设场景
# =============================================================================

PRESETS = {
    "8gb-single": {
        "gpus": [GPU(idx=0, total_gib=8, label="4060Ti-8G")],
        "services": [
            ServiceRequest("funasr", 20.0, "all"),
            ServiceRequest("cosyvoice", 0.3, "clone"),
        ],
        "desc": "8GB 单卡, 20路 ASR + 极低 TTS 需求 (用 funasr, qwen3-asr 装不下)",
    },
    "4090-single": {
        "gpus": [GPU(idx=0, total_gib=24, label="4090-24G")],
        "services": [
            ServiceRequest("qwen3-asr", 5.0, "default"),
            ServiceRequest("cosyvoice", 0.5, "clone"),
        ],
        "desc": "单张 4090 24G, qwen3-asr ~5 req/s + cosyvoice 0.5 req/s",
    },
    "4090-dual": {
        "gpus": [
            GPU(idx=0, total_gib=24, label="4090-24G"),
            GPU(idx=1, total_gib=24, label="4090-24G"),
        ],
        "services": [
            ServiceRequest("qwen3-asr", 20.0, "default"),
            ServiceRequest("cosyvoice", 1.0, "clone"),
        ],
        "desc": "双 4090, 20 路 ASR (qwen3-asr) + ~1 req/s TTS",
    },
    "4090-quad": {
        "gpus": [
            GPU(idx=i, total_gib=24, label="4090-24G") for i in range(4)
        ],
        "services": [
            ServiceRequest("funasr", 24.0, "all"),
            ServiceRequest("qwen3-asr", 10.0, "default"),
            ServiceRequest("cosyvoice", 0.6, "clone"),
        ],
        "desc": "4 张 4090: 24 路 funasr + 10 路 qwen3-asr + 0.6 路 TTS",
    },
    "high-load": {
        "gpus": [
            GPU(idx=i, total_gib=24, label="4090-24G") for i in range(8)
        ],
        "services": [
            ServiceRequest("qwen3-asr", 20.0, "default"),
            ServiceRequest("cosyvoice", 1.0, "clone"),
        ],
        "desc": "8 张 4090: 20 路 qwen3-asr + 1 路 TTS (剩余卡留给冗余)",
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
    print("--- 步骤 2/2: 启用哪些子服务, 每个目标 QPS 多少? ---")
    print("(QPS = 每秒请求数, 流式 WS 也按 '同时活跃的连接数 × 每秒尝试请求' 估)")
    print()
    services = []

    if _ask_yn("启用 funasr (中文 ASR, 适合短句高 QPS)?", default=True):
        qps = _ask_float("  目标 QPS", default=20, min_val=0)
        all_modes = _ask_yn("  需要流式 ASR 吗? (否=offline only, 省显存)",
                           default=True)
        services.append(ServiceRequest(
            "funasr", qps, "all" if all_modes else "offline"
        ))

    if _ask_yn("启用 qwen3-asr (多语种/带标点 ASR, 显存大)?", default=False):
        qps = _ask_float("  目标 QPS", default=5, min_val=0)
        services.append(ServiceRequest("qwen3-asr", qps, "default"))

    if _ask_yn("启用 dolphin (多语种 ASR, 轻量)?", default=False):
        qps = _ask_float("  目标 QPS", default=10, min_val=0)
        services.append(ServiceRequest("dolphin", qps, "default"))

    if _ask_yn("启用 cosyvoice (TTS)?", default=True):
        qps = _ask_float("  目标 QPS (注意: 单副本只 ~0.34 req/s)",
                        default=0.5, min_val=0)
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
            services.append(ServiceRequest("cosyvoice", qps, mode))

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

    placements, warnings = plan(gpus, services)
    print(render(gpus, services, placements, warnings))


if __name__ == "__main__":
    main()
