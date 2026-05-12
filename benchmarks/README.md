# `benchmarks/` — 微服务化重构并发实测

本目录是 `refactor/microservices-vllm` 分支并发模型重构 (`asyncio.to_thread` + 信号量/锁) 的实证材料。所有数据都来自一张 **NVIDIA RTX 4090 24 GB (idx=7)**, 通过裸跑各子服务的 Docker 容器直接打 HTTP/WS 端口得到, 不经过网关, 排除网关侧任何干扰。

```
benchmarks/
├── README.md              ← 本文件
├── audio/                 ← TTS 生成的测试音频 (git-lfs)
│   ├── sample_00.wav .. sample_15.wav
│   └── long_concat.wav    ← 前 8 段拼接, 用于长推理 /health 验证
├── scripts/               ← 测试脚本
│   ├── bench_tts.py            (TTS 并发吞吐)
│   ├── bench_asr.py            (ASR 并发吞吐, funasr / qwen3-asr 通用)
│   ├── bench_event_loop.py     (TTS event-loop 阻塞验证)
│   ├── bench_asr_event_loop.py (ASR event-loop 阻塞验证)
│   ├── bench_cohabit.py        (cosyvoice 同卡多副本 RTF, 实测 0.85 折扣值)
│   ├── bench_voice_crud.py     (voice CRUD 并发安全性)
│   └── gen_audio.py            (用 cosyvoice 生成测试音频)
└── results/               ← 结果 (json + log, json 走 git-lfs)
    ├── tts/               ← cosyvoice 单副本 baseline vs patched
    ├── tts_cohabit/       ← cosyvoice 同卡 1/2/3 副本 RTF 对比
    ├── asr_funasr/        ← funasr 子服务
    ├── asr_qwen3/         ← qwen3-asr-vllm 子服务
    └── voice_crud/        ← voice CRUD 并发测试
```

## 1. 测试方法

每个子服务**单独**跑两个版本对比:

- **baseline**: 重构前的镜像 (handler 内同步推理调用 → 阻塞 event loop), tag 为 `funspeech/<svc>:baseline`
- **patched**: 重构后的镜像 (`asyncio.to_thread` + 并发控制), tag 为 `funspeech/<svc>:patched`

所有容器都用如下命令绑到 GPU 7:

```bash
docker run -d --name <bench> --gpus '"device=7"' -p <port>:<port> \
  --ulimit nofile=65535:65535 \
  -e INTERNAL_SERVICE_TOKEN=funspeech-internal \
  -v "$HOME/.cache/modelscope/hub/models:/root/.cache/modelscope/hub" \
  funspeech/<svc>:<baseline|patched>
```

qwen3-asr 额外需要:

```
-e QWEN3_ASR_MODEL_ID=/root/.cache/modelscope/hub/Qwen/Qwen3-ASR-1.7B
-e QWEN3_ASR_GPU_MEM=0.7
-e HF_HUB_OFFLINE=1
-e TRANSFORMERS_OFFLINE=1
```

cosyvoice 额外需要 (要 mount voices 卷):

```
-e TTS_MODEL_MODE=all
-e CLONE_MODEL_VERSION=cosyvoice3
-e GPU_INFERENCE_CONCURRENCY=2  ← 历史测试遗留, 现版本已硬编码无需设置
-v "$PWD/voices:/app/voices"
```

## 2. 脚本说明

所有脚本接受 `NO_PROXY=* no_proxy=*` 关闭宿主代理 (避免 httpx 误读 SOCKS 代理变量)。

### `scripts/bench_tts.py` — TTS 并发吞吐

```bash
env NO_PROXY="*" no_proxy="*" python3 benchmarks/scripts/bench_tts.py \
  --base-url http://127.0.0.1:18004 \
  --token funspeech-internal \
  --voice "中文女" \
  --concurrencies 1,2,4,8 \
  --out benchmarks/results/tts/xxx.json
```

每个并发档位的工作流:

1. 预热 1 个请求 (让 wetext / clone 模型一阶段就绪)
2. 同时发 N 个 `POST /tts/file`, 用同一个 `httpx.AsyncClient(Limits(...))` 复用连接
3. 统计 `wall` (整批耗时) / `req/s` (= ok / wall) / `p50, p95, max` (单条耗时)

`rtf = 总音频秒数 / wall`: 大于 1 表示比实时快。

### `scripts/bench_asr.py` — ASR 离线并发吞吐

```bash
env NO_PROXY="*" no_proxy="*" python3 benchmarks/scripts/bench_asr.py \
  --base-url http://127.0.0.1:18001 \  # 或 18003 for qwen3
  --token funspeech-internal \
  --concurrencies 1,2,4,8,16,32 \
  --out benchmarks/results/asr_xxx/xxx.json
```

每个档位: 预热 1 条 → 并发 N 条 `POST /asr/file` (multipart) → 统计同 TTS。
音频从 `benchmarks/audio/sample_*.wav` 轮转取 (16 段, 每段 ~4 秒, 22050 Hz)。

### `scripts/bench_event_loop.py` (TTS) / `scripts/bench_asr_event_loop.py` (ASR)

**最关键**的测试 — 验证修复有没有真的解放 event loop。

工作流:

1. 持续以 50 ms / 20 ms 的频率打 `GET /health`, 记录每次耗时
2. 同时发 N 个推理请求
3. 统计推理期间 `/health` 的延迟分布

判读规则:

- baseline: GPU 推理在 `async def` handler 里同步跑 → event loop 被卡 → `/health` 延迟会跳到秒级, 甚至 probe 都打不出去几次
- patched: 推理在 `asyncio.to_thread` 工作线程里 → event loop 自由 → `/health` 延迟 < 50 ms

### `scripts/gen_audio.py` — 用 cosyvoice 生成测试音频

```bash
env NO_PROXY="*" no_proxy="*" BASE_URL=http://127.0.0.1:18004 \
  python3 benchmarks/scripts/gen_audio.py
```

串行调 patched cosyvoice 16 次, 输出 `audio/sample_00..15.wav`。`audio/long_concat.wav` 是后续用 Python `wave` 模块手动拼接前 8 段, 用于 ASR long-form /health 测试。

## 3. 数据文件含义

### JSON 格式 (所有 `*.json`)

```json
{
  "runs": [
    {
      "concurrency": 8,           // 本档位并发数
      "wall_sec": 0.66,           // 整批耗时
      "ok": 8, "failed": 0,
      "req_per_sec": 12.06,       // 吞吐
      "audio_sec_total": 33.0,    // 仅 TTS, 总合成秒数
      "rtf": 1.42,                // 仅 TTS
      "p50": 0.43, "p95": 0.66,   // 单条耗时分位
      "max": 0.66, "min": 0.39, "mean": 0.55,
      "errors": []
    },
    ...
  ]
}
```

### Log 格式 (所有 `*.log`)

bench_*.py 跑时的 stdout 拷贝, 每行一档结果, 人眼快速看。

### Event-loop log (`results/**/event_loop_*.log`)

```
ASR 期间 /health 延迟 (n-asr=8):
  ASR 开始前 (idle): n=9 mean=2.0ms p50=1.8ms p95=2.7ms max=2.7ms
  ASR 进行中       : n=18 mean=311.2ms p50=2.6ms p95=4272.1ms max=4272.1ms
ASR 单条耗时:
  idx=0  elapsed=15.71s  status=200
  ...
```

- `n=` 是 probe 实际采到的样本数 — baseline 这里通常很少 (probe 被卡住自己也打不出请求), patched 接近理论值 (wall_sec ÷ 0.02s)
- `p95` 是核心指标: 推理期间 /health 应该 < 50ms

## 4. 实测结果摘要 (单卡 4090 24G, idx=7)

### TTS (cosyvoice, ~3.5 秒推理)

#### 单副本 baseline vs patched (sem=2 甜蜜点)

| 并发 | baseline wall | patched sem=2 wall | patched 提升 |
|---|---|---|---|
| 1 | 3.67 s | 3.48 s | ±0% |
| 2 | 6.71 s | 5.43 s | **-19%** |
| 4 | 14.47 s | 11.73 s | **-19%** |
| 8 | 28.69 s | 23.46 s | **-18%** |

`req/s` 从 0.28 → 0.34, **吞吐 +21%**。sem=8 反而比 baseline 慢一倍 (GPU 上下文切换), 实测确认 sem=2 是单卡甜蜜点。

#### TTS 实时容量 (RTF, 单副本)

RTF (Real-Time Factor) = 推理耗时 / 生成音频时长。RTF ≤ 1 = 实时, > 1 = 客户端会等。

| N (并发) | 单客户端 RTF | 系统总吞吐 (sys_rtf) | 解读 |
|---|---|---|---|
| 1 | 0.69 | 1.42 | 性能过剩 |
| **2** | **1.05** | **1.42** | **刚好实时, 推荐工作点** |
| 4 | 1.72 | 1.42 | 单客户端开始卡顿 |
| 8 | 3.0 | 1.43 | 严重卡顿 |

**单副本实时容量 = 2 路** (RTF ≤ 1.1)。想 N 路实时 TTS → 至少 ceil(N/2) 副本。

#### TTS 同卡多副本 RTF (新增, 见 `results/tts_cohabit/`)

为什么测? 验证"同卡多副本会不会因为 GPU 抢占变慢"。

| 副本数 (同卡) | sys_rtf 系统总吞吐 | per_client RTF @ N=副本×2 | 结论 |
|---|---|---|---|
| 1 | 1.42 | 1.10 (N=2) | 基线 |
| **2** | **2.43** (+0.85×) | 1.15 (N=4) | 接近线性, 仍踩实时线 |
| 3 | 3.28 (+0.85×) | 1.30 (N=6) | 略超 1, 但仍可用 |

**结论**: 同卡多副本是有效的扩容手段, **每多 1 个同卡副本贡献 ~0.85 单副本容量** (实测 1.42 → 2.43 → 3.28)。`scripts/plan_deployment.py` 因此放开了"同服务不同卡"的硬约束 — 显存够就允许同卡放, 优先用空闲卡。

> qwen3-asr 例外: vLLM 启动直接预占 0.85 × 卡显存, 两个 vLLM 进程同卡会 OOM, 仍强制不同卡。

### TTS event loop 测试 (N=2 并发)

| | baseline | patched |
|---|---|---|
| /health p95 | **1630 ms** | **3.5 ms** |
| /health 样本 | 6 个 (probe 被卡) | 209 个 |

### FunASR (~80ms 推理)

| 并发 | baseline | patched sem=1 | patched sem=4 |
|---|---|---|---|
| req/s @ N=8 | 13.5 | 12.1 | 9.1 |
| req/s @ N=32 | 13.3 | 12.6 | 9.9 |

funasr 单次推理已经吃满 GPU, sem 调大反而变慢。**推荐 sem=1**, throughput 与 baseline 持平, 但 event loop 解放。

### FunASR event loop (N=8 长音频)

| | baseline | patched |
|---|---|---|
| /health p95 | **189 ms** | **18 ms** |
| /health max | **377 ms** | **24 ms** |
| 样本数 | 24 | 65 |

### qwen3-asr-vllm (~190ms 推理, vLLM 引擎)

> **重要**: 最初我把 sem 默认设成 128, 导致多线程同时调 vLLM Python 入口出现死锁 (N≥2 时大量 180s 超时)。修复方案: `Semaphore(128)` → `asyncio.Lock()`, 详见 `services/qwen3_asr_vllm/server.py:157-186` 注释。

| 并发 | baseline | patched |
|---|---|---|
| req/s @ N=32 | 5.02 | 5.04 |
| req/s @ N=64 | 4.56 | **5.26 (+15%)** |

### qwen3-asr event loop (N=8)

| | baseline | patched |
|---|---|---|
| /health p95 | **4272 ms** | **4.5 ms** |
| /health 样本 | 18 个 | 529 个 |
| ASR 单条 min~max | 6.6 ~ 15.7 s | 2.7 ~ 11.9 s |

/health 延迟改善 ~950×。

## 5. 跨子服务对比 (patched 版, 单卡 4090)

| 维度 | funasr | qwen3-asr | cosyvoice |
|---|---|---|---|
| 单条 N=1 延迟 | **80 ms** | 190 ms | 3500 ms |
| 单卡容量 | **~12 req/s** | ~5 req/s (vLLM batch 64 内部并行) | **2 路实时 TTS** (RTF≤1.1) |
| 显存 | 2.5 GB | **13 GB** (含 KV cache) | 3.5 GB |
| 自带标点 | ✗ (需 PUNC 模型) | ✓ | n/a |
| 自带语种识别 | ✗ | ✓ | n/a |
| 长音频质量 | 中文好, 长尾差 | **更好, 跨语种, 方言** | n/a |
| 内部 batching | 无 | **vLLM continuous batching** | 无 |
| 同卡多副本 | OK | ❌ vLLM OOM | OK, 0.85× 系数 |

**funasr 适合**: 高 QPS、纯中文短句、对延迟敏感、显存预算紧
**qwen3-asr 适合**: 标点/语种/混合语种/方言质量优先, 中等 QPS, 显存富余

横向扩展: 都通过 `docker compose up -d --scale <svc>=N` + 网关 `_HttpReplicaPool` 自动均衡。

## 6. 复跑指南

```bash
# 1) 关闭宿主 SOCKS 代理 (httpx 会误识别)
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

# 2) 起子服务 (示例: patched cosyvoice)
docker run -d --name funspeech-cosyvoice-bench --gpus '"device=7"' -p 18004:8004 \
  --ulimit nofile=65535:65535 \
  -e PORT=8004 -e TTS_MODEL_MODE=all -e TTS_DEVICE=cuda:0 \
  -e CLONE_MODEL_VERSION=cosyvoice3 \
  -e INTERNAL_SERVICE_TOKEN=funspeech-internal \
  -v "$HOME/.cache/modelscope/hub/models:/root/.cache/modelscope/hub" \
  -v "$PWD/voices:/app/voices" \
  funspeech/cosyvoice:patched

# 3) 等 ready
until docker logs funspeech-cosyvoice-bench 2>&1 | grep -q "Application startup complete"; do
  sleep 5
done

# 4) 跑测试
env NO_PROXY="*" no_proxy="*" python3 benchmarks/scripts/bench_tts.py \
  --base-url http://127.0.0.1:18004 \
  --concurrencies 1,2,4,8 \
  --out benchmarks/results/tts/tts_my_run.json

env NO_PROXY="*" no_proxy="*" python3 benchmarks/scripts/bench_event_loop.py \
  --base-url http://127.0.0.1:18004 --n-tts 2 \
  | tee benchmarks/results/tts/event_loop_my_run.log

# 5) 清理
docker stop funspeech-cosyvoice-bench && docker rm funspeech-cosyvoice-bench
```

## 7. 已知边界

- 所有 patched 数据都在 **sem 的合适值** 下取得 (`cosyvoice=2`, `funasr=1`, `qwen3-asr=Lock`)。盲目调大 sem 会触发 GPU 上下文切换或 vLLM 线程不安全。
- 测试样本只有 16 段, 都来自 cosyvoice 合成 (中文女声, 22050 Hz, 4-7 秒/段)。真实业务负载下数字可能有 ±20% 波动。
- 单卡测试无法体现横向扩展收益。要测多 GPU 多副本的真并发, 把 `funspeech/<svc>:patched` 起多个实例 + 网关 `*_SERVICE_URLS` 写多个 URL 即可。
