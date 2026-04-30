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

## 架构

```
                       ┌─────────────────────────┐
   外部客户端 ────────► │  gateway (CPU)          │
   (HTTP/WS)           │  - Aliyun/OpenAI 协议   │
                       │  - 句子状态机 + ITN     │
                       │  - 采样率/格式转换      │
                       └────────┬────────────────┘
                                │ HTTP / WS (X-Internal-Token)
                ┌───────────────┼───────────────┬───────────────┐
                ▼               ▼               ▼               ▼
         funasr (GPU)   dolphin (GPU)   qwen3-asr (GPU)   cosyvoice (GPU)
         Paraformer/    DataoceanAI     Qwen3-ASR-1.7B    CosyVoice2/3
         SenseVoice     Dolphin Small   (vLLM 加速)        (in-process)
```

子服务各自一个 `pyproject.toml` + `uv.lock` + Dockerfile,依赖完全隔离。
例如 funasr 用 transformers 4.51.3,qwen3-asr 用 transformers 4.57.1 + vLLM 0.11+,
互不冲突。

## 快速开始

### 1. 准备环境

```bash
git clone <your-repo>
cd agents-e3557b6358

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
docker compose up -d                         # 默认: gateway + funasr + cosyvoice
docker compose --profile dolphin up -d       # 加上 dolphin
docker compose --profile qwen3-asr up -d     # 加上 qwen3-asr-vllm
docker compose --profile dolphin --profile qwen3-asr up -d   # 全开
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

# TTS
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
| funasr-0 | 8001 | funspeech/funasr | ✅ | ✅ | (默认) |
| dolphin-0 | 8002 | funspeech/dolphin | ✅ | ❌ | `dolphin` |
| qwen3-asr-0 | 8003 | funspeech/qwen3-asr | ✅ | ❌ | `qwen3-asr` |
| cosyvoice-0 | 8004 | funspeech/cosyvoice | ✅ | ✅ | (默认) |

每个子服务暴露 `GET /health` + 自有业务端点(详见各 `services/*/README.md`)。

## 对外 API(网关)

### ASR

| 端点 | 方法 | 说明 |
|---|---|---|
| `/stream/v1/asr` | POST | 一句话语音识别 |
| `/stream/v1/asr/models` | GET | 模型列表 |
| `/stream/v1/asr/health` | GET | 健康检查 |
| `/ws/v1/asr` | WS | 流式识别(Aliyun 协议) |

可识别模型(`models.json`):`paraformer-large`、`sensevoice-small`、`dolphin-small`、`qwen3-asr-flash`(后两者需要起对应 profile)。

### TTS

| 端点 | 方法 | 说明 |
|---|---|---|
| `/stream/v1/tts` | POST | 语音合成 |
| `/openai/v1/audio/speech` | POST | OpenAI 兼容 |
| `/rest/v1/tts/async` | POST/GET | 异步长文本合成 |
| `/stream/v1/tts/voices` | GET | 音色列表 |
| `/stream/v1/tts/voices/info` | GET | 音色详细信息 |
| `/stream/v1/tts/voices/refresh` | POST | 刷新音色 |
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
| `TTS_MODEL_MODE` | `all` | `all` / `sft` / `clone` |
| `ASR_ENABLE_REALTIME_PUNC` | `false` | 流式中间结果是否带标点 |
| `AUTO_LOAD_CUSTOM_ASR_MODELS` | - | 启动时预热的额外 ASR 模型 id |
| `FUNASR_SERVICE_URLS` | `http://funasr-0:8001` | 子服务 URL,逗号分隔多副本 |
| `DOLPHIN_SERVICE_URLS` | `http://dolphin-0:8002` | |
| `QWEN3_ASR_SERVICE_URLS` | `http://qwen3-asr-0:8003` | |
| `COSYVOICE_SERVICE_URLS` | `http://cosyvoice-0:8004` | |
| `SERVICE_REQUEST_TIMEOUT` | `60` | 子服务调用超时(秒) |

子服务专属 env(模型版本、TRT/FP16/vLLM 等)请见 `.env.example` 和各 `services/*/README.md`。

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

## 模型权重缓存

所有 GPU 子服务通过 bind mount 共享 `MODELSCOPE_CACHE`(默认 `~/.cache/modelscope`),
因此即便 funasr / cosyvoice 各自的 transformers 版本不同,权重文件可以复用。

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

## 音色管理(克隆)

零样本克隆音色由 cosyvoice 子服务托管。把 `张三.wav` + `张三.txt` 放到 `./voices/` 卷,然后:

```bash
curl -X POST http://localhost:8000/stream/v1/tts/voices/refresh
```

或通过 API 上传(需要外部 multipart 接入,见 docs/)。所有音色状态(`spk2info.pt` + `voice_registry.json`)
持久化到 `./voices/` 卷。

## 已知设计取舍

- **vLLM 加速 CosyVoice 默认关闭**:vLLM 0.11+ 要求 transformers ≥4.55,
  与 CosyVoice 主代码所需的 4.51.3 冲突。如果性能瓶颈明显,可拆出 `services/cosyvoice_vllm/`
  独立 venv 启用,见 `services/cosyvoice/README.md` 的多副本注释。
- **Qwen3-ASR 流式不走 vLLM 通用 `/v1/realtime`**:vLLM 通用 realtime 端点对
  Qwen3-ASR 质量明显劣化(无跨段上下文、无 token 修订,见 vllm Issue #35767)。
  我们用官方 `qwen_asr.Qwen3ASRModel.streaming_transcribe` + `init_streaming_state`,
  在子服务进程内跑,具备 unfixed_chunk / unfixed_token 修订能力。
- **音色 CRUD 单副本**:`/voices/*` 写操作必须打到 cosyvoice-0(否则状态分裂)。
  网关侧的 sticky 路由暂未实现,生产多副本部署需自行加 LB 规则或独立的写副本。

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
