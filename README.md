<div align="center">

![FunSpeech](./docs/images/banner.png)

  <h3>开箱即用的本地私有化部署语音服务 — 微服务架构</h3>

ASR + TTS API 网关,兼容阿里云语音 API 与 OpenAI TTS API,支持 WebSocket 流式协议。
模型推理由独立子服务承载(每个引擎独立 venv + 容器),通过 docker-compose 编排。

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/CUDA-12.1+-%2376B900?logo=nvidia&logoColor=white)
![Static Badge](https://img.shields.io/badge/uv-0.11+-%23DE5F87)

</div>

> [!IMPORTANT]
> **从单体版升级? 先读** [`docs/migration_to_latest.md`](./docs/migration_to_latest.md)
>
> 本分支已重构为微服务架构 (gateway + 4 个 GPU 子服务)。对外 HTTP/WS 协议**字节级兼容**, 客户端代码不用动; 但部署侧 docker-compose、模型缓存挂载路径、`.env` 变量都有变化。迁移文档覆盖:数据原地复用、必改的 mount 路径、`.env` 增删项、零停机切换、回滚步骤。

## 架构

```
                       ┌─────────────────────────┐
   外部客户端 ────────► │  gateway (CPU)          │
   (HTTP/WS)           │  - Aliyun/OpenAI 协议   │
                       │  - 句子状态机 + ITN     │
                       │  - 采样率/格式转换      │
                       └────────┬────────────────┘
                                │ HTTP / WS (X-Internal-Token)
                ┌───────────────┼───────────────┬───────────────┬───────────────┐
                ▼               ▼               ▼               ▼               ▼
         funasr (GPU)   dolphin (GPU)   qwen3-asr (GPU)   cosyvoice (GPU)   qwen3-tts (GPU)
         Paraformer/    DataoceanAI     Qwen3-ASR-1.7B    CosyVoice2/3     Qwen3-TTS
         SenseVoice     Dolphin Small   (vLLM 加速)        (in-process)     local
```

子服务各自一个 `pyproject.toml` + `uv.lock` + Dockerfile,依赖完全隔离。
例如 funasr 用 transformers 4.51.3,qwen3-asr 用 transformers 4.57.1 + vLLM 0.11+,
互不冲突。

## 快速开始

### 1. 准备环境

```bash
git clone https://cnb.cool/nexa/FunSpeech.git
cd FunSpeech

# 必须: 拉 cosyvoice 上游源码 (git submodule)
# 不做这一步, cosyvoice 镜像启动会 ModuleNotFoundError
git submodule update --init --recursive

cp .env.example .env  # 按需修改
```

如果机器在国内,把代理放进 `.env`:

```bash
HTTP_PROXY=http://host.docker.internal:7890   # macOS/Windows Docker Desktop
HTTPS_PROXY=http://host.docker.internal:7890
# Linux 服务器: 改成宿主机 LAN IP, 例如 http://192.168.1.10:7890
```

### 2. 构建 + 启动

Dockerfile 用了 `RUN --mount=type=cache` 给 apt / uv 双缓存,需要 BuildKit。Docker 23+ 默认开启;低版本手动启用:

```bash
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
```

```bash
docker compose build                         # 重 build 时 apt/uv 走本机缓存,极快
docker compose up -d                         # 默认: gateway + qwen3-asr + cosyvoice
docker compose --profile funasr up -d        # 加上 funasr (paraformer/sensevoice)
docker compose --profile dolphin up -d       # 加上 dolphin
TTS_ENGINE=qwen3-tts docker compose up -d gateway qwen3-asr-0 qwen3-tts-0  # 改用 Qwen3-TTS Base Clone
TTS_ENGINE=qwen3-tts-vllm-omni docker compose up -d gateway qwen3-asr-0 qwen3-tts-vllm-omni-0
TTS_ENGINE=cosyvoice3-vllm-omni docker compose up -d gateway qwen3-asr-0 cosyvoice3-vllm-omni-0
docker compose --profile funasr --profile dolphin up -d   # 全部 ASR 引擎
```

服务暴露在 `http://localhost:${GATEWAY_PORT:-8000}`。

### 3. 验证

```bash
# 健康
curl http://localhost:8000/stream/v1/asr/health
curl http://localhost:8000/stream/v1/tts/health

# ASR
curl -X POST "http://localhost:8000/stream/v1/asr?format=wav&sample_rate=16000" \
     -H "Content-Type: application/octet-stream" \
     --data-binary @audio.wav

# TTS (默认 CosyVoice 可直接用预设音色; Qwen3-TTS 需要先注册 clone voice)
curl -X POST "http://localhost:8000/stream/v1/tts" \
     -H "Content-Type: application/json" \
     -d '{"text":"你好","voice":"中文女"}' \
     --output speech.wav
```

WebSocket 测试页:

- ASR: `http://localhost:8000/ws/v1/asr/test`
- TTS: `http://localhost:8000/ws/v1/tts/test`

## 服务列表

| 服务 | 端口 | 镜像 | GPU | 默认启动 | profile |
|---|---|---|---|---|---|
| gateway | 8000 | funspeech/gateway | ❌ | ✅ | (默认) |
| funasr-0 | 8001 | funspeech/funasr | ✅ | ❌ | `funasr` |
| dolphin-0 | 8002 | funspeech/dolphin | ✅ | ❌ | `dolphin` |
| qwen3-asr-0 | 8003 | funspeech/qwen3-asr | ✅ | ✅ | (默认, 默认 ASR 引擎) |
| cosyvoice-0 | 8004 | funspeech/cosyvoice | ✅ | ✅ | (默认) |
| qwen3-tts-0 | 8005 | funspeech/qwen3-tts | ✅ | ❌ | `qwen3-tts` |
| qwen3-tts-vllm-omni-0 | 8006 | funspeech/qwen3-tts-vllm-omni | ✅ | ❌ | `qwen3-tts-vllm-omni` |
| cosyvoice3-vllm-omni-0 | 8007 | funspeech/cosyvoice3-vllm-omni | ✅ | ❌ | `cosyvoice3-vllm-omni` |

网关一次只选择一个 TTS 后端。默认是 `TTS_ENGINE=cosyvoice`;改用任一非默认 TTS 时显式启动 `gateway qwen3-asr-0 <tts-service>`，避免把默认 CosyVoice 一起拉起。legacy Qwen3-TTS 的 clone 音色持久化在 `./qwen3_voices`;vLLM-Omni Qwen3-TTS 和 CosyVoice3 分别使用 `./qwen3_omni_voices`、`./cosyvoice3_omni_voices`。4090 实测 Qwen3-TTS vLLM-Omni 能把 Base Clone 提到 2 req/s 以上,但约需 19-20 GiB 净显存;CosyVoice3 vLLM-Omni 当前吞吐接近默认 CosyVoice3,主要用于 vLLM-Omni 兼容验证和后续调优。

每个子服务暴露 `GET /health` + 自有业务端点(详见各 `services/*/README.md`)。

## 对外 API(网关)

### ASR

| 端点 | 方法 | 说明 |
|---|---|---|
| `/stream/v1/asr` | POST | 一句话语音识别 |
| `/stream/v1/asr/models` | GET | 模型列表 |
| `/stream/v1/asr/health` | GET | 健康检查 |
| `/ws/v1/asr` | WS | 流式识别(Aliyun 协议) |

可识别模型(`models.json`):`qwen3-asr-flash`(默认)、`paraformer-large`、`sensevoice-small`(后两者需 `--profile funasr`)、`dolphin-small`(需 `--profile dolphin`)。

### TTS

| 端点 | 方法 | 说明 |
|---|---|---|
| `/stream/v1/tts` | POST | 语音合成 |
| `/openai/v1/audio/speech` | POST | OpenAI 兼容 |
| `/rest/v1/tts/async` | POST/GET | 异步长文本合成 |
| `/stream/v1/tts/voices` | GET | 音色列表 |
| `/stream/v1/tts/voices/info` | GET | 音色详细信息 |
| `/stream/v1/tts/voices/refresh` | POST | 刷新网关音色列表缓存; 扫描目录请调对应子服务 `/voices/refresh` |
| `/stream/v1/tts/health` | GET | 健康检查 |
| `/ws/v1/tts` | WS | 双向流式合成(Aliyun 协议) |

外部协议与之前的进程内版本完全兼容。

## 配置(网关侧 env)

| 变量 | 默认 | 说明 |
|---|---|---|
| `GATEWAY_PORT` | `8000` | 对外暴露端口 |
| `APPTOKEN` / `APPKEY` | - | 外部鉴权(可选) |
| `INTERNAL_SERVICE_TOKEN` | `funspeech-internal` | 网关→子服务鉴权头(`X-Internal-Token`) |
| `ASR_MODEL_MODE` | `all` | `all` / `offline` / `realtime` |
| `TTS_ENGINE` | `cosyvoice` | `cosyvoice` / `qwen3-tts` / `qwen3-tts-vllm-omni` / `cosyvoice3-vllm-omni`;选择 TTS 后端,同一网关只能选一个 |
| `TTS_MODEL_MODE` | `all` | `all` / `sft` / `clone` |
| `ASR_ENABLE_REALTIME_PUNC` | `false` | 流式中间结果是否带标点 |
| `AUTO_LOAD_CUSTOM_ASR_MODELS` | - | 启动时预热的额外 ASR 模型 id |
| `FUNASR_SERVICE_URLS` | `http://funasr-0:8001` | 子服务 URL,逗号分隔多副本 |
| `DOLPHIN_SERVICE_URLS` | `http://dolphin-0:8002` | |
| `QWEN3_ASR_SERVICE_URLS` | `http://qwen3-asr-0:8003` | |
| `COSYVOICE_SERVICE_URLS` | `http://cosyvoice-0:8004` | |
| `QWEN3_TTS_SERVICE_URLS` | `http://qwen3-tts-0:8005` | `TTS_ENGINE=qwen3-tts` 时的本地开源 Qwen3-TTS Base Clone 子服务 |
| `QWEN3_TTS_VLLM_OMNI_SERVICE_URLS` | `http://qwen3-tts-vllm-omni-0:8006` | `TTS_ENGINE=qwen3-tts-vllm-omni` 时的 vLLM-Omni Speech API 子服务 |
| `COSYVOICE3_VLLM_OMNI_SERVICE_URLS` | `http://cosyvoice3-vllm-omni-0:8007` | `TTS_ENGINE=cosyvoice3-vllm-omni` 时的 vLLM-Omni Speech API 子服务 |
| `SERVICE_REQUEST_TIMEOUT` | `60` | 子服务调用超时(秒) |
| `INFERENCE_THREAD_POOL_SIZE` | `max(4, CPU 核数)` | 网关同步调用线程池大小;高 QPS 调大 |
| `HTTPX_MAX_CONNECTIONS` | `200` | 网关→子服务 HTTP 连接池上限;>100 req/s 时调到 500+ |
| `HTTPX_MAX_KEEPALIVE` | `50` | 保活连接数上限 |

子服务专属 env(模型版本、TRT/FP16/vLLM 等)请见 `.env.example`、`docs/deployment.md §6` 和各 `services/*/README.md`。

> 关于"GPU 并发数":子服务内部的 GPU 并发已在代码里硬编码 (funasr/dolphin=1, cosyvoice=2, qwen3-asr=Lock+vLLM 内部 batching), **不通过环境变量暴露**。横向扩展请用多副本, 见下文。

## 开发

### 单独跑某个子服务

```bash
cd services/funasr
uv sync
PORT=8001 INTERNAL_SERVICE_TOKEN=test uv run python server.py
```

### 网关本地跑(连容器里的子服务)

```bash
uv sync
FUNASR_SERVICE_URLS=http://localhost:8001 \
COSYVOICE_SERVICE_URLS=http://localhost:8004 \
INTERNAL_SERVICE_TOKEN=funspeech-internal \
uv run python start.py
```

## 性能与副本规划

在 NVIDIA RTX 4090 24G 上的实测单副本容量 (完整数据见 [`benchmarks/`](./benchmarks/README.md)):

| 子服务 | 单副本容量 | 单条延迟 | 显存 |
|---|---|---|---|
| funasr (all) | **~12 req/s** | ~80 ms | ~3 GiB |
| dolphin | ~12 req/s | ~80 ms | ~1 GiB |
| qwen3-asr | **~5 req/s** (vLLM 内部 batch 可吃 64 并发) | ~190 ms | **0.85 × 卡显存** (vLLM KV pool) |
| cosyvoice (clone) | **2 路实时 TTS** (RTF ≈ 1.05) | ~3.5 s / 句 | ~4 GiB |

> ASR 看 req/s, TTS 看"几路实时" (RTF ≤ 1) — 单副本 sem=2 时同时 2 路 RTF=1.05 刚好实时, 4 路就开始 RTF=1.75 卡顿。想 N 路实时 TTS 需 ceil(N/2) 张卡。

**横向扩展 = 多卡多副本**, 不是单卡多副本 (同一服务两副本绑同一张卡, 实测总容量不升反降)。

用规划脚本一键算出应该开几副本 / 怎么绑卡, 并在当前目录生成完整 `docker-compose.generated.yml`:

```bash
python3 scripts/plan_deployment.py                  # 交互式
python3 scripts/plan_deployment.py --preset 4090-quad   # 看预设
python3 scripts/plan_deployment.py --list-presets
```

详细说明见 [`docs/deployment.md §4.3`](./docs/deployment.md)。

## 模型权重缓存

GPU 子服务使用两类权重缓存:

| 缓存 | 默认宿主机路径 | 容器路径 | 主要使用者 |
|---|---|---|---|
| `MODELSCOPE_CACHE` | `~/.cache/modelscope/hub/models` | `/root/.cache/modelscope/hub` | FunASR / CosyVoice / Qwen3-ASR |
| `HF_CACHE` | `~/.cache/huggingface` | `/root/.cache/huggingface` | Qwen3-TTS |

因此即便各子服务 transformers / qwen 包版本不同,权重文件也可以跨容器重用。

提前下载:

```bash
pip install modelscope
modelscope download --model iic/CosyVoice-300M-SFT
modelscope download --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
# 按需:
modelscope download --model DataoceanAI/dolphin-small
modelscope download --model Qwen/Qwen3-ASR-1.7B
```

Qwen3-TTS Base 预下载:

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

## 音色管理(克隆)

CosyVoice 和 Qwen3-TTS 使用不同的音色目录,不要混用:

| TTS 后端 | 目录 | 模型语义 | 说明 |
|---|---|---|---|
| `cosyvoice` | `./voices` | 预设音色 + clone | `TTS_MODEL_MODE=all/clone` 时支持 clone |
| `qwen3-tts` | `./qwen3_voices` | Base Clone only | 没有 `中文女` 这类预设音色,合成前必须先注册 clone voice |

CosyVoice:把 `张三.wav` + `张三.txt` 放到 `./voices/` 卷,然后直接调用 cosyvoice 子服务扫描:

```bash
curl -X POST \
  -H "X-Internal-Token: ${INTERNAL_SERVICE_TOKEN:-funspeech-internal}" \
  http://localhost:8004/voices/refresh
```

Qwen3-TTS Base Clone:直接上传参考音频和准确参考文本到 qwen3-tts 子服务:

```bash
curl -F name=zhangsan \
  -F prompt_text='参考音频对应的准确文本' \
  -F audio=@./ref.wav \
  -H "X-Internal-Token: ${INTERNAL_SERVICE_TOKEN:-funspeech-internal}" \
  http://localhost:8005/voices
```

之后合成时使用注册名:

```bash
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","voice":"zhangsan"}' \
  --output speech.wav
```

所有音色状态会持久化到各自目录:CosyVoice 是 `spk2info.pt` + `voice_registry.json`,Qwen3-TTS 是参考音频、`prompts/*.pt` 和 `voice_registry.json`。

## 已知设计取舍

- **vLLM 加速 CosyVoice 默认关闭**:vLLM 0.11+ 要求 transformers ≥4.55,
  与 CosyVoice 主代码所需的 4.51.3 冲突。如果性能瓶颈明显,可拆出 `services/cosyvoice_vllm/`
  独立 venv 启用,见 `services/cosyvoice/README.md` 的多副本注释。
- **Qwen3-ASR 流式不走 vLLM 通用 `/v1/realtime`**:vLLM 通用 realtime 端点对
  Qwen3-ASR 质量明显劣化(无跨段上下文、无 token 修订,见 vllm Issue #35767)。
  我们用官方 `qwen_asr.Qwen3ASRModel.streaming_transcribe` + `init_streaming_state`,
  在子服务进程内跑,具备 unfixed_chunk / unfixed_token 修订能力。
- **Qwen3-ASR 子服务进程内 GPU 串行**: vLLM 的 Python `LLM(...)` 入口
  非线程安全, 子服务用 `asyncio.Lock` 保证只有一个调用进 vLLM。这**不影响** vLLM
  自身的 continuous batching (batching 在 engine 内部), 但要靠多副本扩吞吐。
- **音色 CRUD 多副本同步**: URL 列表第一个 = 写副本 (primary)。通过网关 TTS engine 的
  `voice_manager` 写入时会自动路由到 primary,写后广播 `POST /voices/reload`。
  直接调用子服务接口时请打到 primary。详见 [`docs/deployment.md §5.2`](./docs/deployment.md)。

## 目录结构

```
.
├── app/                       # 网关代码
│   ├── api/v1/                # FastAPI 路由 (Aliyun + OpenAI)
│   ├── services/asr/          # ASR 引擎抽象 + HTTP 客户端
│   ├── services/tts/          # TTS 引擎抽象 + HTTP 客户端 (含 voice_manager)
│   ├── services/websocket_*.py
│   ├── core/config.py         # 网关 env
│   └── utils/audio.py         # 重采样、PCM/WAV 转码、ITN
├── services/                  # 各子服务(独立 venv + Dockerfile)
│   ├── funasr/
│   ├── dolphin/
│   ├── qwen3_asr_vllm/
│   └── cosyvoice/             # third_party/CosyVoice 是官方 submodule
├── scripts/
│   ├── plan_deployment.py     # 副本规划 (零依赖)
│   └── analyze_audio_rms.py   # 远场过滤阈值分析
├── benchmarks/                # 4090 实测数据 + 测试脚本 (git-lfs)
│   ├── README.md
│   ├── scripts/               # bench_tts.py / bench_asr.py / ...
│   ├── audio/                 # TTS 生成的测试样本
│   └── results/               # 结果 (json + log)
├── docker-compose.yml
├── .env.example
└── pyproject.toml             # 网关 deps (CPU only, ~57 包)
```

## 相关链接

- [CosyVoice (FunAudioLLM)](https://github.com/FunAudioLLM/CosyVoice)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [Dolphin (DataoceanAI)](https://github.com/DataoceanAI/Dolphin)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [vLLM](https://github.com/vllm-project/vllm)

## 许可证

MIT — 见 [LICENSE](LICENSE)。
