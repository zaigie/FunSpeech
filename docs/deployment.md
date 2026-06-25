# 部署指南

完整使用流程见根 [README.md](../README.md)。本文聚焦在生产部署细节:服务编排、GPU/显存调度、各子服务可调环境变量。

## 一、前置要求

- Docker ≥ 24
- Docker Compose v2(随 Docker Desktop 自带,Linux 用 `docker compose` 命令)
- NVIDIA Container Toolkit(GPU 子服务必需)

```bash
# Ubuntu / Debian 安装 NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 二、构建前置准备

### 2.1 拉子模块 (cosyvoice 必需)

`services/cosyvoice/third_party/CosyVoice` 是上游官方 git submodule。**首次 clone 后必须执行**:

```bash
git submodule update --init --recursive
```

否则 cosyvoice 镜像里 `third_party/CosyVoice` 是空目录,容器启动 `from cosyvoice.cli.cosyvoice import ...` 会立刻 `ModuleNotFoundError`。

### 2.2 启用 BuildKit

Dockerfile 用 `RUN --mount=type=cache,target=/var/cache/apt` 与 `--mount=type=cache,target=/root/.cache/uv` 给 apt 与 uv 加缓存挂载,大幅缩短重复 build 时间。Docker 23+ 默认开启 BuildKit;低版本手动:

```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

### 2.3 构建期 HTTP 代理

国内拉 PyPI / HuggingFace 慢,推荐配置代理:

```bash
# .env
HTTP_PROXY=http://host.docker.internal:7890
HTTPS_PROXY=http://host.docker.internal:7890
```

- macOS / Windows Docker Desktop:`host.docker.internal` 直接可用
- Linux 服务器:换成宿主机 LAN IP(例如 `192.168.1.10`),或在 docker daemon 里配 default proxy

代理仅用于构建期,Dockerfile 末尾会清空运行期 ENV。

## 三、启动方式

```bash
docker compose build              # 首次或 Dockerfile 变更后
docker compose up -d              # 默认: gateway + qwen3-asr + cosyvoice

docker compose --profile funasr up -d                              # 加 funasr (paraformer/sensevoice)
docker compose --profile dolphin up -d                             # 加 dolphin
docker compose --profile funasr --profile dolphin up -d            # 全部 ASR 引擎
```

并行加速 build (子服务镜像之间互相独立):

```bash
docker compose build --parallel
```

> 首次 build 内存可能吃紧 (vLLM 编译 + torch wheel),如果 OOM 退到串行 build。

## 四、服务编排与 GPU 拓扑

### 4.1 服务列表

| 服务 | 端口 | GPU | 默认启动 | profile |
|---|---|---|---|---|
| `gateway` | 8000 | ❌ | ✅ | (默认) |
| `funasr-0` | 8001 | ✅ | ❌ | `funasr` |
| `dolphin-0` | 8002 | ✅ | ❌ | `dolphin` |
| `qwen3-asr-0` | 8003 | ✅ | ✅ | (默认, 默认 ASR 引擎) |
| `cosyvoice-0` | 8004 | ✅ | ✅ | (默认) |
| `qwen3-tts-0` | 8005 | ✅ | ❌ | `qwen3-tts` |

每个 GPU 子服务通过 `deploy.resources.reservations.devices[].device_ids` 选择宿主机 GPU。Docker/NVIDIA runtime 只把对应卡透传进容器后,容器内 GPU 会从 `0` 重新编号,所以单卡容器里的 `CUDA_VISIBLE_DEVICES` 应保持 `"0"` 或不设置。

### 4.2 资源占用 (基于 4090 24G 实测)

**单副本容量和显存** — 全部数据来自 `benchmarks/results/`:

ASR 用并发 / 吞吐衡量, TTS 用 RTF (推理耗时 ÷ 音频时长, ≤1 为实时) 衡量。

| 子服务 | 单副本容量 | 单条延迟 | 权重显存 | 实际占卡 |
|---|---|---|---|---|
| `funasr-0` (all) | **~12 req/s**, sem=1 单 GPU 占用串行 | 70-80 ms | ~2.5-3 GiB | ~3 GiB |
| `funasr-0` (offline) | ~12 req/s | ~80 ms | ~0.7 GiB | ~1 GiB |
| `dolphin-0` | ~12 req/s | ~80 ms | ~0.6 GiB | ~1 GiB |
| `qwen3-asr-0` | **~5 req/s** (单连接), vLLM 内部 batch 可吃 64 并发 | ~190 ms | ~4 GiB **权重** | **总 = `QWEN3_ASR_GPU_MEM × 卡显存`** |
| `cosyvoice-0` (clone) | **同时 2 路实时 TTS (RTF≈1.05)**, 第 3 路开始 RTF>1 卡顿 | ~3.5 s / 句 | ~3.5 GiB | ~4 GiB |
| `cosyvoice-0` (all, sft+clone) | 同上, 2 路实时 | ~3.5 s | ~5 GiB | ~5.5 GiB |

> **TTS 容量的实际意义**: 用户最关心的不是 "TTS req/s", 而是 "我能同时开几路 TTS 让客户端听起来都流畅"。
> 实测 (`benchmarks/results/tts/tts_patched_sem2.json`):
> - N=1 路: RTF=0.69 (推理快于实时 40%, 性能过剩)
> - **N=2 路: RTF=1.05** (刚好实时, 推荐工作点)
> - N=4 路: RTF=1.75 (单路被拉长 75%, 流式会卡)
> - N=8 路: RTF=3.0 (严重卡顿)
> 因此 cosyvoice 单副本的"实时容量" = **2 路**, 想要 N 路实时 TTS 就要 ceil(N/2) 副本。

> **qwen3-asr 显存特殊**: 一启动就**直接预占** `QWEN3_ASR_GPU_MEM × 卡显存` (vLLM 把权重 + KV cache 池都放在这一片里), 不是只占权重 4 GiB。这是和 funasr / cosyvoice 最不一样的地方。

**关键约束**:
1. **同一个服务的多个副本不能放同一张卡** — 它们会抢同一份 GPU SM, 实际吞吐 ≈ 1 副本。横向扩展 = 多卡多副本, 不是单卡多副本。
2. **不同服务可以共置同一张卡** — 例如 funasr (3 GiB) + cosyvoice (5 GiB) 可以同卡, 共占 ~9 GiB。
3. **qwen3-asr 占卡霸道**: 独占时 `QWEN3_ASR_GPU_MEM=0.85` 性能最好但吃掉 ~20 GiB; 共置时降到 0.3-0.4 (~7-10 GiB) 但 KV pool 小, 高 QPS 下批处理空间小。

### 4.3 怎么算自己应该几副本几卡: 用脚本

不要靠拍脑袋, 直接用 `scripts/plan_deployment.py`:

```bash
# 交互式: 一步步问你 GPU + 目标并发 / 实时路数
python3 scripts/plan_deployment.py

# 调整 ASR 排队容忍 (默认 1.0s, 调大副本变少, 客户端等更久)
python3 scripts/plan_deployment.py --max-queue-sec 2.0

# 用预设场景看看
python3 scripts/plan_deployment.py --list-presets
python3 scripts/plan_deployment.py --preset 4090-quad

# 从 JSON 文件读 (适合 CI / 复用)
python3 scripts/plan_deployment.py --json my_setup.json
```

**脚本输入**: GPU 列表 (每张卡的显存) + 启用哪些服务 + 每个服务的容量需求:
- ASR (funasr / dolphin / qwen3-asr): **目标并发数** (同时活跃的客户端数)
- TTS (cosyvoice): **想同时支持几路实时 TTS** (按 RTF ≤ 1 计, 单副本 = 2 路)

**脚本输出**:
- 每个服务需要几副本 (ASR 按 max_queue_sec 反推, TTS 按 RTF 实时容量反推)
- 每个副本绑哪张卡 (worst-fit 启发, 让显存均匀分布)
- 显存预算表 (含 vLLM KV pool 估算)
- 当前目录下的完整 `docker-compose.generated.yml`
  (如已存在则写成 `docker-compose.generated.1.yml`、`.2.yml` 等, 不覆盖)
- 哪些目标因卡数/显存不够达不到 → 给出明确的扩容建议

生成后先做静态检查, 再用生成文件启动:

```bash
docker compose -f docker-compose.generated.yml config
docker compose -f docker-compose.generated.yml up -d

# 如果规划里包含 funasr / dolphin, 按脚本最后打印的命令带上 profile:
docker compose -f docker-compose.generated.yml --profile funasr up -d
```

**典型例子** (摘自脚本输出):

| 场景 | GPU | 建议 |
|---|---|---|
| 单 24G 卡: qwen3-asr 20 并发 + cosyvoice 2 路实时 | 1× 4090 | qwen3-asr 占满卡 0 (KV pool 0.85), cosyvoice 放不下 → 降 `QWEN3_ASR_GPU_MEM=0.5` 或加卡 |
| 双 24G 卡: qwen3-asr 40 并发 + cosyvoice 4 路实时 | 2× 4090 | qwen3-asr 各占 1 张卡, cosyvoice 装不下 → 加第 3 张卡 |
| 4× 24G 卡: dolphin 10 并发 + qwen3-asr 20 并发 + cosyvoice 4 路实时 | 4× 4090 | qwen3-asr#0 占卡 0, cosyvoice 2 副本占卡 1/2 (4 路实时), dolphin 1 副本搭卡 1 |

> **ASR 与 TTS 副本数差异巨大的原因**: ASR 单条几十~两百毫秒, 1 秒钟单副本能轮十几遍, 高并发只是排队问题; TTS 单条 3.5s 长任务, 想 20 路实时就必须 10 个 GPU 同时算, 这是硬性物理限制, 加卡就是加卡。

### 4.4 跨卡分布的两种方式

**方式 A — 全部贴卡 0(默认):** `docker-compose.yml` 里所有 GPU 子服务使用 `CUDA_VISIBLE_DEVICES: "0"` 和 `count: 1`,适合单卡或想让所有模型共置。

**方式 B — 跨卡分布:** 直接运行 `plan_deployment.py`, 用生成的完整 `docker-compose.generated.yml` 启动。这个文件不依赖 override merge, 新副本会带齐 `image`、`volumes`、`healthcheck`、`restart`、`ulimits`、`profiles` 等字段。

注意 `device_ids: ["0", "1"]` 不会在子服务进程内启用张量并行(那需要 vLLM 的 `tensor_parallel_size`,且只对装不下的大模型有意义,本项目模型都是 0.5B–1.7B,装得下,不需要切)。多卡的正确用法是**多副本**(§5)。

## 五、多副本

### 5.1 添加副本

**推荐**: 用 `scripts/plan_deployment.py` 一键生成完整 `docker-compose.generated.yml` (见 §4.3)。

**手动**: 每张额外的卡 = 一个新副本。不要只写 `environment` + `deploy`: 新服务不会继承 `funasr-0` / `cosyvoice-0` 的字段。手写时应从生成的 `docker-compose.generated.yml` 或基础服务复制完整服务定义, 再改:

- `service` 名称与 `container_name` (例如 `funasr-1` / `funspeech-funasr-1`)
- `deploy.resources.reservations.devices[].device_ids` 使用宿主机卡号
- `CUDA_VISIBLE_DEVICES` 保持 `"0"` (容器内重新编号后的第一张卡)
- 网关 `*_SERVICE_URLS` 列表, 例如 `FUNASR_SERVICE_URLS=http://funasr-0:8001,http://funasr-1:8001`

`funasr` / `dolphin` 新副本还要保留对应 `profiles`, 否则会从可选服务变成默认启动服务。

网关侧 `_HttpReplicaPool` 会做最少连接 + 随机选副本调度;每个外部 WS 会话**绑定固定副本**(session 级亲和性),保证 cache 状态不串。

> ⚠️ **同一服务的多个副本不要绑同一张宿主机卡**: 例如不要 `funasr-0` 和 `funasr-1` 的 `device_ids` 都写同一个宿主卡号。两个副本会抢同一份 GPU SM, 总吞吐反而下降 (实测见 `benchmarks/results/`)。 横向扩展 = 多卡多副本。

### 5.2 cosyvoice 多副本的写副本与同步

`spk2info.pt` 是被各副本加载到内存里的 Python dict, **磁盘共享 ≠ 内存一致**:

- 副本 A 加音色 Alice → `torch.save()` 整个内存 dict 到磁盘
- 同时副本 B 加 Bob → 它的内存里没 Alice → save 时把 Alice **覆盖掉**
- 即使串行写, 副本 B 内存里仍然没有 Alice, 收到合成请求会失败

**网关已实现的同步机制(自 commit `<本次提交>` 起):**

1. URL 列表第一个 = 主写副本 (primary)。所有 `POST /voices`、`DELETE /voices/{name}`、`POST /voices/refresh` 自动路由到 primary, 不再走副本池随机调度。
2. 写完成后, 网关向其它副本广播 `POST /voices/reload`, 它们从磁盘热重载 `spk2info.pt` + `voice_registry.json`, 不重启进程。
3. 广播失败仅打 warning, 不抛异常 — 保证写本身始终成功; 失败副本最坏到下次广播或重启才同步。
4. 合成请求 (`POST /tts/file`、`WS /tts/stream`) 仍走副本池, 任意副本都行。

部署时 **`COSYVOICE_SERVICE_URLS` 第一个 URL 就是 primary**:

```bash
# cosyvoice-0 写, cosyvoice-1 / cosyvoice-2 只读但能合成
COSYVOICE_SERVICE_URLS=http://cosyvoice-0:8004,http://cosyvoice-1:8004,http://cosyvoice-2:8004
```

主副本宕了怎么办: 当前不会自动 failover, 需手动调换 URL 列表顺序并重启网关。

### 5.3 `/voices/reload` 端点

cosyvoice 子服务暴露 `POST /voices/reload`, 用磁盘上的 `spk2info.pt` + `voice_registry.json` **覆盖**内存状态(磁盘没有的 zero-shot 音色会从内存里删掉, 但预设音色不动)。

| 场景 | 用法 |
|---|---|
| 网关写后自动广播 | 见 §5.2, 无需手动调用 |
| 直接更新文件后手动同步 | `curl -X POST -H "X-Internal-Token: $INTERNAL_SERVICE_TOKEN" http://cosyvoice-0:8004/voices/reload` |
| 排查不一致 | 任意副本调一次, 返回 `{clone_voices, registry_voices, clone_loaded}` |

仅 `clone_loaded=true` 的副本(即 `TTS_MODEL_MODE` 包含 `clone`)才会真正重载 `spk2info`; `sft` only 副本只重载 registry。

## 六、子服务环境变量参考

所有 GPU 子服务共有的:

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | 子服务各自约定(8001/2/3/4) | 监听端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `INTERNAL_SERVICE_TOKEN` | `funspeech-internal` | 网关→子服务鉴权头(`X-Internal-Token`) |
| `MODELSCOPE_PATH` | `~/.cache/modelscope/hub` | 容器内权重缓存路径(已通过 bind mount 共享) |
| `CUDA_VISIBLE_DEVICES` | `0` | 容器内 CUDA 可见卡序号。用 compose `device_ids` 单卡透传时保持 `0`;不要写宿主机卡号 |

下面分子服务列出独有的 env。

### 6.1 funasr-0 (端口 8001,profile=funasr)

| 变量 | 默认 | 说明 |
|---|---|---|
| `ASR_MODEL_MODE` | `all` | `all` / `offline` / `realtime`,决定预加载哪个 paraformer |
| `ASR_DEVICE` | `auto` | `auto` / `cpu` / `cuda:0` 等,与 `CUDA_VISIBLE_DEVICES` 配合 |
| `FUNASR_OFFLINE_MODEL` | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 离线 paraformer 模型 id |
| `FUNASR_REALTIME_MODEL` | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | 流式 paraformer |
| `VAD_MODEL` | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | VAD 模型 id |
| `VAD_MODEL_REVISION` | `v2.0.4` | |
| `PUNC_MODEL` | `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch` | 离线标点 |
| `PUNC_MODEL_REVISION` | `v2.0.4` | |
| `PUNC_REALTIME_MODEL` | `iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727` | 实时标点 |

显存 (4090 实测):`offline` 模式只占 ~0.7 GB;`all` 模式约 2.5–3 GB(两个 paraformer + VAD + 双 PUNC)。单副本吞吐 ~12 req/s (单条 70-80ms), 高 QPS 用多副本扩。

### 6.2 dolphin-0 (端口 8002,profile=dolphin)

| 变量 | 默认 | 说明 |
|---|---|---|
| `DOLPHIN_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0`。跨宿主卡部署时由 compose `device_ids` 控制宿主卡,容器内仍用可见卡 `0` |
| `DOLPHIN_SIZE` | `small` | dolphin 模型规模(目前仅 `small`) |
| `DOLPHIN_MODEL_PATH` | `DataoceanAI/dolphin-small` | 模型 id (相对 `MODELSCOPE_PATH`) |

显存约 0.6 GB,可与 funasr 同卡共置。

### 6.3 qwen3-asr-0 (端口 8003,默认启动)

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWEN3_ASR_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | vLLM 加载的模型 id (也可指向本地路径) |
| `QWEN3_ASR_GPU_MEM` | `0.8` | **vLLM 启动时直接预留这么大比例的卡显存做权重+KV pool** |
| `QWEN3_ASR_MAX_NEW_TOKENS` | `4096` | 单步生成上限 (官方推荐, 影响 KV pool 大小) |
| `QWEN3_ASR_MAX_BATCH` | `128` | vLLM 内部 continuous batching 最大 batch size |
| `QWEN3_UNFIXED_CHUNK_NUM` | `2` | 流式状态机 unfixed chunk 数(token 修订窗口) |
| `QWEN3_UNFIXED_TOKEN_NUM` | `5` | 流式状态机 unfixed token 数 |
| `QWEN3_CHUNK_SIZE_SEC` | `2.0` | 流式 chunk 时长(秒) |

**`QWEN3_ASR_GPU_MEM` 是显存调度最关键的 env:**

| 卡显存 | 与其它服务同卡共置 | 独占整张卡 |
|---|---|---|
| 8 GB | 不建议 (模型权重 4 GiB + KV pool 起步 1 GiB 就已经挤兑其它服务) | `0.85`(~6.8 GB,够 1.7B 模型 + 小 KV) |
| 16 GB | `0.4`(预留 ~9.6 GB 给别的) | `0.85` |
| 24 GB | `0.3`(预留 ~16 GB 给别的, 但 KV pool 小, 高并发会等位) | **`0.85`** (实测最佳, ~5 req/s) |
| 40+ GB | `0.2`–`0.3` 即可 | `0.85` |

`gpu_memory_utilization` 越大 KV cache 池越大,vLLM 能并行处理的请求越多;但子服务进程内的 vLLM 入口是**串行**调用 (Python LLM 接口非线程安全, 见 `services/qwen3_asr_vllm/server.py:156` 的 `_get_vllm_lock` 实现注释), batching 在 vLLM 引擎内部完成, 实际收益要看 KV pool 容量 + 请求长短。

> 关于子服务的"GPU 并发" — 我们曾尝试在子服务 handler 层加 `asyncio.Semaphore(N)` 让多个推理同时进 GPU, 实测**完全负优化** (vLLM 死锁 / torch 模型上下文切换变慢)。现在每个子服务的 GPU 并发数都在代码里**硬编码**, 不通过环境变量暴露 — 想加并发请用多副本 (§5)。

### 6.4 cosyvoice-0 (端口 8004)

| 变量 | 默认 | 说明 |
|---|---|---|
| `TTS_MODEL_MODE` | `all` | `all` / `sft` / `clone`,决定预加载预设音色模型还是克隆模型 |
| `TTS_DEVICE` | `auto` | `auto` / `cpu` / `cuda:0` |
| `CLONE_MODEL_VERSION` | `cosyvoice3` | `cosyvoice2` / `cosyvoice3` |
| `SFT_MODEL_ID` | `iic/CosyVoice-300M-SFT` | SFT (预设音色) 模型 id |
| `CLONE_MODEL_ID` | `iic/CosyVoice2-0.5B` | CosyVoice2 克隆模型 id (仅 `CLONE_MODEL_VERSION=cosyvoice2` 时用) |
| `COSYVOICE3_MODEL_ID` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | CosyVoice3 模型 id |
| `TTS_LOAD_TRT` | `false` | TensorRT 加速 flow / vocoder |
| `TTS_ENABLE_FP16` | `false` | FP16 推理。CosyVoice3 + FP16 + TRT 同时开存在 NaN 风险 |
| `TTS_LOAD_VLLM` | `false` | **进程内 vLLM 加速 LLM 段。vLLM 0.11+ 要求 transformers ≥4.55,与 CosyVoice 主代码 4.51.3 冲突,默认关闭** |
| `VOICES_DIR` | `/app/voices` | 容器内音色目录(挂载 `./voices`) |

**显存 (4090 实测):**

- `clone` only(默认 CosyVoice3):~3.5 GB
- `sft` only:~1.5 GB
- `all`(双模型同时加载):~5 GB
- 启用 TRT:加 ~0.5–1 GB
- 启用 vLLM:加 ~1 GB(LLM 段被 vLLM 接管,有自己的 KV cache)

**实时容量** (推荐用这个指标, 而不是 req/s):

| 指标 | 单副本 (sem=2, 4090) | 实际含义 |
|---|---|---|
| **同时实时 TTS 路数** | **2 路** (RTF ≈ 1.05) | 推荐工作点, 客户端流式播放不卡 |
| 1 路时的 RTF | ~0.69 | 单客户端: 推理 3.48s 生成 5s 音频, 性能过剩 |
| 4 路时的 RTF | ~1.75 | 单路被拉长, 流式播放开始卡顿 |
| 8 路时的 RTF | ~3.0 | 严重卡顿 |
| 离线吞吐 | ~0.34 req/s | 整段合成场景的 req/s, 跟"实时路数"是同一物理量的两种说法 |

CosyVoice 是 autoregressive 解码器, 单条音频本身就要 GPU 跑几秒, 这是硬性的物理限制。
**想要 N 路实时 TTS → ceil(N/2) 副本 → ceil(N/2) 张 GPU**。10 路实时要 5 张卡, 20 路要 10 张卡。

### 6.5 qwen3-tts-0 (端口 8005)

Qwen3-TTS 走开源本地 `qwen-tts` 包,网关侧通过 `TTS_ENGINE=qwen3-tts` 单选切换。当前集成默认只支持 `Qwen/Qwen3-TTS-12Hz-0.6B-Base` 的 Base Clone: 添加音色时上传参考音频和参考文本,合成时使用已注册的 clone voice。

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWEN3_TTS_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 当前网关按 Base Clone 语义集成 |
| `QWEN3_TTS_DEVICE` | `cuda:0` | 容器内 GPU 编号 |
| `QWEN3_TTS_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `QWEN3_TTS_ATTN_IMPLEMENTATION` | `sdpa` | 可按环境改成 `flash_attention_2` |
| `QWEN3_TTS_LANGUAGE` | `Auto` | `Auto` 会传 `None`,由模型处理 |
| `QWEN3_TTS_VOICES_DIR` | `/app/qwen3_voices` | Qwen3 clone 音色目录; compose 默认挂载 `./qwen3_voices` |
| `QWEN3_TTS_X_VECTOR_ONLY_MODE` | `false` | `false` 使用 ICL clone,添加音色必须提供准确参考文本;`true` 仅使用 speaker embedding |
| `HF_ENDPOINT` | 空 | Hugging Face 镜像端点,国内可设 `https://hf-mirror.com` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | 空 | 离线部署时设为 `1` |

Qwen3-TTS 使用独立的 `./qwen3_voices` 目录,不复用 CosyVoice 的 `./voices`。和 CosyVoice 一样,只有 clone 模型加载成功时才会接受 `/voices` 写入;非 Base 模型会拒绝音色 CRUD。

请用 `python3 scripts/plan_deployment.py` 实算 (脚本会问你"想同时支持多少路实时 TTS")。

## 七、网关环境变量参考

| 变量 | 默认 | 说明 |
|---|---|---|
| `GATEWAY_PORT` | `8000` | 对外暴露端口 |
| `WORKERS` | `1` | uvicorn worker 进程数;>1 时每个 worker 独立加载客户端 |
| `INFERENCE_THREAD_POOL_SIZE` | `max(4, CPU 核数)` | 网关内部派发同步阻塞调用的线程池大小 |
| `APPTOKEN` / `APPKEY` | - | 外部鉴权(可选,见 §八) |
| `INTERNAL_SERVICE_TOKEN` | `funspeech-internal` | 必须与子服务一致 |
| `SERVICE_REQUEST_TIMEOUT` | `60` | 网关→子服务调用超时(秒) |
| `SERVICE_HEALTHCHECK_INTERVAL` | `5` | 健康状态缓存窗口(秒) |
| `HTTPX_MAX_CONNECTIONS` | `200` | 网关→子服务 HTTP 客户端的连接池上限。高 QPS (>100 req/s) 可调到 500+ |
| `HTTPX_MAX_KEEPALIVE` | `50` | 保活连接数上限 |
| `FUNASR_SERVICE_URLS` | `http://funasr-0:8001` | 逗号分隔多副本 |
| `DOLPHIN_SERVICE_URLS` | `http://dolphin-0:8002` | |
| `QWEN3_ASR_SERVICE_URLS` | `http://qwen3-asr-0:8003` | |
| `COSYVOICE_SERVICE_URLS` | `http://cosyvoice-0:8004` | |
| `QWEN3_TTS_SERVICE_URLS` | `http://qwen3-tts-0:8005` | |
| `TTS_ENGINE` | `cosyvoice` | `cosyvoice` / `qwen3-tts`;选择 TTS 后端,同一网关只能选一个 |
| `ASR_MODEL_MODE` | `all` | 仅影响 `models.json` 兼容性校验,真正模式由 funasr 子服务决定 |
| `TTS_MODEL_MODE` | `all` | 影响 `get_voices()` 返回过滤 |
| `ASR_ENABLE_REALTIME_PUNC` | `false` | 流式中间结果是否带标点(转发给 funasr 子服务) |
| `AUTO_LOAD_CUSTOM_ASR_MODELS` | - | 启动时预热的额外 ASR 模型 id,逗号分隔 |
| `ASR_ENABLE_NEARFIELD_FILTER` | `true` | 网关侧远场过滤开关 |
| `ASR_NEARFIELD_RMS_THRESHOLD` | `0.01` | RMS 阈值 |
| `ASR_NEARFIELD_FILTER_LOG_ENABLED` | `true` | 过滤命中是否打日志 |

## 八、鉴权

- **外部鉴权**(可选):设置 `APPTOKEN` / `APPKEY`,客户端通过
  `X-NLS-Token`(Aliyun 接口)或 `Authorization: Bearer xxx`(OpenAI 接口)携带
- **内部鉴权**(必备):`INTERNAL_SERVICE_TOKEN` 网关→子服务通过
  `X-Internal-Token` 头携带。**生产环境务必改默认值**

## 九、数据卷

| 主机路径 | 容器路径 | 用途 | 注意 |
|---|---|---|---|
| `${MODELSCOPE_CACHE}` (默认 `~/.cache/modelscope/hub/models`) | `/root/.cache/modelscope/hub` | 模型权重缓存,所有 GPU 子服务共享 | 见下方说明 |
| `./voices` | `/app/voices`(只在 cosyvoice) | 零样本克隆音色 + spk2info.pt | 持久化用户数据 |
| `./temp` | `/app/temp`(只在 gateway) | 网关临时音频文件 | gateway 返回 FileResponse 后会 BackgroundTask 自动删除 |
| `./data` | `/app/data`(只在 gateway) | 异步 TTS 任务库 | |
| `./logs` | `/app/logs`(只在 gateway) | 日志 | |

### 9.1 模型缓存目录映射 (重要)

宿主机 `~/.cache/modelscope/hub/models/` **直接** mount 成容器内 `/root/.cache/modelscope/hub/`。这样做的原因是新旧 modelscope 版本的目录布局不同:

- 新版 (≥1.30): `hub/models/<org>/<model>` (例如 `hub/models/Qwen/Qwen3-ASR-1.7B`)
- 旧版 (=1.20, 各子服务用的版本): `hub/<org>/<model>` (例如 `hub/Qwen/Qwen3-ASR-1.7B`)

把宿主机的 `models/` mount 成容器的 `hub/`, 两边路径就都对上了。

**首次启动**: 子服务会从 modelscope 自动下载所需权重 (具体见各服务 `*_MODEL_ID` env)。提前手动下到 `~/.cache/modelscope/hub/models/<org>/<model>` 可以避开首次启动数分钟的等待。

**离线部署 (无外网)**: 把 `~/.cache/modelscope/hub/models/` 整个 rsync 到目标机, mount 进去即可。**注意 qwen3-asr** 离线时还要把 `QWEN3_ASR_MODEL_ID` 改成容器内本地路径,例如 `/root/.cache/modelscope/hub/Qwen/Qwen3-ASR-1.7B`,并设置 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`;否则 vLLM 会按 Hugging Face repo id 查缓存快照并报 `LocalEntryNotFoundError`。

## 十、健康检查与启动顺序

每个子服务都有 `/health`,docker-compose 配了 `healthcheck`:

- funasr / dolphin: `start_period=180s`,等模型加载
- qwen3-asr / cosyvoice: `start_period=300s`,vLLM 与 CosyVoice 加载更慢

`gateway.depends_on` 用 `condition: service_started`,即使子服务还没就绪网关也能起来对外报 503,避免一处异常全栈不可用。改 `service_healthy` 即可严格依赖。

## 十一、排错

```bash
# 看子服务状态
docker compose ps

# 看子服务日志
docker compose logs -f funasr-0
docker compose logs -f qwen3-asr-0    # vLLM 日志在这里

# 进容器查环境
docker compose exec gateway env | grep -E "SERVICE_URLS|TOKEN"
docker compose exec qwen3-asr-0 env | grep -E "QWEN3|CUDA"

# 网关侧能不能连到子服务
docker compose exec gateway curl -fsS http://funasr-0:8001/health
docker compose exec gateway curl -fsS http://qwen3-asr-0:8003/health

# 看显存占用
nvidia-smi
docker compose exec qwen3-asr-0 nvidia-smi   # 看容器视角(通常重新编号为 GPU 0)

# 重启某个服务(不影响其它)
docker compose restart funasr-0

# 完整清理(注意会丢容器,保留卷)
docker compose down

# 连卷一起清(会丢音色数据!)
docker compose down -v
```

### 常见问题

- **`out of memory` / `CUDA OOM`**:多半是 qwen3-asr 与别的服务同卡共置但 `QWEN3_ASR_GPU_MEM` 太大,降到 `0.3`–`0.4` 再试。
- **网关一直报 503,但子服务日志看着正常**:看 `docker compose ps` 是否 `(healthy)`;若 healthcheck 还在 `start_period`,等加载完。重构后子服务 `/health` 严格反映模型加载状态:模型没加载完会返回 503 让网关知道。
- **funasr 子服务挂了**:旧版本 `services/funasr/server.py` 缺 import,WS 流式会 NameError——升级到 `0783c3b` 之后的版本即可。
- **CosyVoice3 输出全是噪音**:`TTS_ENABLE_FP16=true` + `TTS_LOAD_TRT=true` 同时开会有 NaN,关一个。
- **不知道几副本几卡才够**: 用 `python3 scripts/plan_deployment.py` (零依赖, 见 §4.3)。
- **多副本副本同 GPU**: 不要这样做。同一服务的两个副本绑同一张卡, 实测总吞吐不升反降 (GPU SM 抢占)。每个副本一张卡。
- **qwen3-asr 启动报 Hugging Face 离线缓存错误**: 离线场景务必把 `QWEN3_ASR_MODEL_ID` 设成本地路径,并设 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`, 见 §9.1。
- **高 QPS 网关 fd 耗尽 / `Too many open files`**: 容器加 `ulimits: { nofile: 65535 }`; 同时检查 `HTTPX_MAX_CONNECTIONS` 是否够大。
