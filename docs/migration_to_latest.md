# 升级指南: 单体版 → 微服务版

如果你已经在生产跑老版本 (单 `funspeech` 容器, 进程内加载 funasr + cosyvoice), 现在想升到本分支 (`gateway` + 4 个 GPU 子服务), 这份文档帮你**快速完成迁移, 不丢生产数据, 不改对外接口**。

读完先记三件事:

1. **对外 HTTP / WebSocket 协议完全兼容** — 客户端代码一行都不用动。具体接口对比见下文 §3。
2. **同一台机器的现有模型缓存可以原地复用** — 不用重下模型, 但 mount 路径要改 (§4.1)。
3. **零样本音色数据原地兼容** — `./voices/` 卷直接 mount 给新版 cosyvoice-0 即可 (§4.2)。

详细的部署细节见 [`deployment.md`](./deployment.md), 本文聚焦"从旧到新"的变化点。

---

## 1. 为什么要升级

| 维度 | 旧版 (单体) | 新版 (微服务) |
|---|---|---|
| 部署形态 | 1 个容器, 进程内 import funasr / cosyvoice | gateway (CPU) + funasr/cosyvoice/qwen3-asr/dolphin 各自容器 (GPU) |
| 依赖冲突 | funasr 与 cosyvoice 的 transformers/torch 版本互掐, 经常需要手动 pin | 每个子服务独立 venv (`uv.lock`), 不互相干扰 |
| 多 GPU 扩展 | 单进程, 多 GPU 也只能轮转 | 多副本横向扩展, 真正能吃满 N 张卡 |
| 新模型 (Qwen3-ASR) | ❌ 无法集成 (vLLM 0.11 与旧版 transformers 冲突) | ✅ 独立子服务, vLLM 加速 |
| GPU 推理期 event loop | 被同步推理阻塞, /health 探针 p95 可达 1.6 秒 | to_thread offload, /health p95 ~5 ms (实测见 `benchmarks/`) |
| 健康检查 | 模型加载失败时仍返回 200 "healthy" | 加载失败返回 503 status=unhealthy, 失败原因 |
| 网关→子服务超时 | n/a (同进程) | WS recv 全部加 timeout, 上游卡死不让客户端 hang |
| 多副本写一致性 | 单进程无此问题 | primary 写副本 + 自动广播 `/voices/reload` |

如果旧版稳定够用、用不上新模型, 不升也行 — 但新提交都在这个分支, 之后只修这边的 bug。

---

## 2. 迁移路线总览

最少改动的"原地升级"路径 (推荐 staging 先跑一遍):

```
旧版容器停下来
  ↓
git pull + git submodule update --init --recursive
  ↓
模型缓存目录复制一份新位置 (或 mv, 见 §4.1)
  ↓
.env 加几个新变量 (见 §5)
  ↓
docker compose build (新的 4 + 1 镜像)
  ↓
docker compose up -d
  ↓
跑 §6 的 smoke 测试
```

零停机迁移 (双轨切流量) 见 §8。

---

## 3. 对外接口兼容性

**结论先行**: HTTP 路由、JSON 字段、WebSocket 消息封装、二进制帧、关闭码全部**字节级**保持不变。审计来自 `app/api/v1/` 和 `app/models/websocket_*.py` 的 diff。

| 端点 | 状态 | 说明 |
|---|---|---|
| `POST /stream/v1/asr` | ✅ 完全一致 | 请求体、响应字段、HTTP code 全部不变 |
| `POST /stream/v1/tts` | ✅ 完全一致 | |
| `POST /openai/v1/audio/speech` | ✅ 完全一致 | OpenAI 兼容协议 |
| `POST/GET /rest/v1/tts/async` | ✅ 完全一致 | 异步长文本合成 |
| `WS /ws/v1/asr` | ✅ 完全一致 | 所有阿里云 ASR 消息 envelope 不变 |
| `WS /ws/v1/tts` | ✅ 完全一致 | 但每个 chunk 之间的 50ms sleep 删了, 客户端会更早收到首帧 |
| `GET /stream/v1/asr/health` | ⚠️ 一处破坏 | **`memory_usage` 字段已删除** (单进程才能算 GPU 占用); `device` 值从 `cuda:0` 变成 `remote:http://funasr-0:8001` |
| `GET /stream/v1/asr/models` | ⚠️ 微调 | `exists` 字段总是返回 `true` (网关已经看不到磁盘上的模型文件) |
| 客户端断连 (WS) | ✅ 一致 | 重构的 cosyvoice 增加了 `client_state` 检查, 客户端断连后子服务不再浪费 GPU |

**清单**: 如果你的客户端代码读 `/asr/health` 的 `memory_usage` 字段, 这个会 KeyError。其他都不用动。

WebSocket 流式 ASR / TTS 客户端**不需要改任何代码**, 包括 `X-NLS-Token` 头、`payload.format`、`payload.sample_rate` 等。

---

## 4. 数据迁移

### 4.1 模型缓存

**旧 mount**:
```yaml
volumes:
  - ~/.cache/modelscope:/root/.cache/modelscope
```

**新 mount** (各 GPU 子服务共享):
```yaml
volumes:
  - ~/.cache/modelscope/hub/models:/root/.cache/modelscope/hub
```

为什么不同: 新版用的 modelscope SDK 把模型下到 `hub/models/<org>/<model>` (新版布局), 但子服务里集成的工具链还有按 `hub/<org>/<model>` (旧版布局) 取模型的。把宿主机 `models/` mount 成容器 `hub/`, 两边都能找到。

**怎么操作**:

```bash
# 看一下你的旧缓存里有什么
ls ~/.cache/modelscope/hub/

# 如果已经有了 models/ 子目录 (新 modelscope 已下过模型), 不用动
ls ~/.cache/modelscope/hub/models/

# 如果没有 models/, 把现有目录搬下去:
mkdir -p ~/.cache/modelscope/hub/models_new
mv ~/.cache/modelscope/hub/* ~/.cache/modelscope/hub/models_new/ 2>/dev/null || true
mv ~/.cache/modelscope/hub/models_new ~/.cache/modelscope/hub/models
```

之后 docker-compose 自动按 `x-modelscope-cache: &modelscope-cache` 这个 anchor mount, 不用手写。

### 4.2 零样本克隆音色

**旧**: `./voices/` mount 给单容器, 里面有 `*.wav`, `*.txt`, `spk2info.pt`, `voice_registry.json`。

**新**: `./voices/` mount 给 `cosyvoice-0` 容器, 路径和内容完全不变。所以**什么都不用做** — 第一次启动新版时, cosyvoice 子服务会从磁盘恢复出全部音色。

> 多副本场景: 第一个 `COSYVOICE_SERVICE_URLS` URL 是写副本 (primary), 所有 `POST/DELETE /voices` 都打到它, 它写完后网关自动广播 `/voices/reload` 给其它副本。详见 `deployment.md §5.2`。

### 4.3 临时文件和数据库

`./temp/`, `./data/`, `./logs/` 这三个 mount 现在只挂给 `gateway`, 内容和路径不变。临时 TTS 文件现在通过 FastAPI `BackgroundTask` 在响应发完后自动删 (老版本是一直泄漏), 这个改动不影响数据迁移。

---

## 5. `.env` 变化点

### 5.1 删 / 改

| 旧 env | 新版去向 |
|---|---|
| `TTS_GPUS=0` / `ASR_GPUS=0` | 删掉。改用每个子服务自己的 `CUDA_VISIBLE_DEVICES`, 默认都是 `"0"`。详见 `deployment.md §4`。 |
| `INFERENCE_THREAD_POOL_SIZE` | 保留, 但**只控制网关线程池**, 不再影响子服务推理。子服务有自己的 GPU 并发常量 (硬编码) |
| `WORKERS` | 保留, 网关 uvicorn worker 数 |

### 5.2 新增

```bash
# 网关→子服务的内部鉴权 (生产务必改默认值)
INTERNAL_SERVICE_TOKEN=funspeech-internal

# 子服务 URL (单机部署默认即可, 多副本时逗号分隔)
FUNASR_SERVICE_URLS=http://funasr-0:8001
COSYVOICE_SERVICE_URLS=http://cosyvoice-0:8004
DOLPHIN_SERVICE_URLS=http://dolphin-0:8002       # 用 --profile dolphin 才生效
QWEN3_ASR_SERVICE_URLS=http://qwen3-asr-0:8003   # 用 --profile qwen3-asr 才生效

# qwen3-asr 离线场景务必加 (有外网就不用)
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# 高 QPS 才需要调
HTTPX_MAX_CONNECTIONS=200         # 网关→子服务 HTTP 连接池上限
HTTPX_MAX_KEEPALIVE=50
SERVICE_REQUEST_TIMEOUT=120       # 网关→子服务的请求超时 (秒)
```

### 5.3 不变

`APPTOKEN`, `APPKEY`, `ASR_MODEL_MODE`, `TTS_MODEL_MODE`, `CLONE_MODEL_VERSION`, `TTS_LOAD_TRT`, `TTS_ENABLE_FP16`, `AUTO_LOAD_CUSTOM_ASR_MODELS`, `ASR_ENABLE_REALTIME_PUNC`, `ASR_ENABLE_NEARFIELD_FILTER`, `ASR_NEARFIELD_RMS_THRESHOLD` — 这些 env 含义都不变, 直接搬过来即可。

> ⚠️ `ASR_MODEL_MODE` / `TTS_MODEL_MODE` 在新版本里实际生效的是**子服务侧**的同名 env (`funasr-0` / `cosyvoice-0` 各自读一份)。网关侧的同名 env 只用来过滤 `models.json` 和 `get_voices()` 返回。docker-compose.yml 默认两边都从一个变量取值, 不用分开配。

---

## 6. 启动与验证

### 6.1 启动

```bash
# 1. 拉子模块 (新增, 不做会 ModuleNotFoundError)
git submodule update --init --recursive

# 2. 构建 (走代理, 国内必须)
HTTP_PROXY=http://host.docker.internal:7890 \
HTTPS_PROXY=http://host.docker.internal:7890 \
docker compose build

# 3. 启动 (默认 = gateway + funasr-0 + cosyvoice-0)
docker compose up -d

# 等子服务模型加载完 (funasr ~30s, cosyvoice ~60s)
docker compose ps    # 等所有都 (healthy)
```

启用 dolphin / qwen3-asr (不是默认的, 要加 profile):

```bash
docker compose --profile dolphin --profile qwen3-asr up -d
```

### 6.2 Smoke 验证

```bash
# 健康
curl http://localhost:8000/stream/v1/asr/health
curl http://localhost:8000/stream/v1/tts/health

# ASR 一句话识别 (跟旧版一模一样的调用)
curl -X POST "http://localhost:8000/stream/v1/asr?format=wav&sample_rate=16000" \
     -H "Content-Type: application/octet-stream" \
     --data-binary @test.wav

# TTS 合成
curl -X POST "http://localhost:8000/stream/v1/tts" \
     -H "Content-Type: application/json" \
     -d '{"text":"测试合成","voice":"中文女"}' \
     --output out.wav

# 看子服务日志 (旧版没这个东西)
docker compose logs -f funasr-0
docker compose logs -f cosyvoice-0
```

---

## 7. 性能容量规划

旧版单进程 GIL 限制下, 单 4090 大约能撑:

- funasr 离线 ASR: ~8-10 req/s (跟 worker 数量绑死)
- cosyvoice TTS: 1-2 路实时 (单进程 + 单 GPU)

新版**单副本**实测 (NVIDIA 4090 24G, 见 `benchmarks/results/`):

| 子服务 | 单副本容量 | 显存 |
|---|---|---|
| funasr (all) | ~12 req/s (单条 ~80ms) | ~3 GiB |
| qwen3-asr | ~5 req/s (单连接), vLLM 内部 batch 可吃 64 并发 | ~20 GiB (vLLM KV pool 0.85) |
| cosyvoice (clone) | **2 路实时 TTS** (RTF ≈ 1.05); 第 3 路开始 RTF > 1 卡顿 | ~4 GiB |

> **TTS 容量看 RTF 而不是 req/s**: TTS 单条要跑 3-4 秒, 用户真正关心的是"能同时开几路 TTS 让客户端听起来不卡"。实测单副本 (sem=2) 同时跑 2 路时 RTF=1.05 (刚好实时), 4 路时 RTF=1.75 (开始卡顿)。**1 张卡 = 2 路实时 TTS**, 想 10 路就要 5 张卡, 没有捷径。

想要更高容量: **横向扩多副本** (多卡)。不要在同一张卡上塞同服务的多副本 — 实测会因 GPU SM 抢占, 总吞吐 / RTF 反而恶化。

具体几副本几卡, 用规划脚本一键算 (零依赖):

```bash
python3 scripts/plan_deployment.py                  # 交互问答
python3 scripts/plan_deployment.py --preset 4090-dual
python3 scripts/plan_deployment.py --list-presets
```

脚本会问:
- ASR (funasr / dolphin / qwen3-asr): 目标**并发数** (同时活跃客户端数)
- TTS (cosyvoice): 想同时支持几路**实时 TTS** (RTF ≤ 1)

输出可直接粘贴的 `docker-compose.override.yml` 和 `.env` 片段。完整说明见 [`deployment.md §4.3`](./deployment.md)。

---

## 8. 零停机切换 (可选)

如果不能停服务, 双轨方案:

1. 新版部署到不同端口 (改 `GATEWAY_PORT=8001`), 与旧版并行运行
2. 用各自的 `MODELSCOPE_CACHE` 和 `voices/` 路径 (避免读写冲突)
3. 在反向代理 (nginx / Caddy / ALB) 上灰度: 5% → 25% → 100%
4. 观察 §7 实测吞吐和 `/stream/v1/asr/health` 健康状态
5. 100% 切完后停旧版, 把 `GATEWAY_PORT` 改回 8000

**注意**: 零样本音色卷不要双写 — 双轨期间所有写 (`POST /voices`) 只走一边, 否则 `spk2info.pt` 内存与磁盘会不一致 (新版的 voice CRUD 加了原子写, 但解决不了并发跨实例写)。

---

## 9. 回滚

新版本完全单独的镜像和卷 mount, 回滚很干净:

```bash
docker compose down
git checkout main           # 或之前的 tag
docker compose up -d        # 旧版会重新拿到 ~/.cache/modelscope (没动过) 和 ./voices/ (兼容)
```

唯一要注意的: 如果在新版上做过音色 CRUD (写到了 `voices/`), 旧版 cosyvoice 启动时会读到这些新音色, **大概率**没问题 (`spk2info.pt` 格式一致), 但极小概率 cosyvoice2/3 之间的 spk 张量字段有差异。担心的话, 切换前 `cp -a voices voices.backup_$(date +%F)`。

---

## 10. 排错速查

| 现象 | 大概率原因 | 解法 |
|---|---|---|
| `git submodule update` 没拉, cosyvoice 启动 ModuleNotFoundError | 漏了 §6.1 第一步 | 重跑 `git submodule update --init --recursive` |
| 网关 503 但子服务日志看着 OK | 健康检查在 `start_period` 内 | 等 funasr 30s / cosyvoice 60s / qwen3-asr 120s |
| `OSError: We couldn't connect to 'https://huggingface.co'` (qwen3-asr) | 离线场景没设 OFFLINE env | `.env` 加 `HF_HUB_OFFLINE=1` `TRANSFORMERS_OFFLINE=1` |
| 音色少了 / 上传后看不到 | 多副本部署但写到了非 primary 副本 | `COSYVOICE_SERVICE_URLS` 第一个 URL 改成你想做主写的 |
| GPU OOM | 多服务同卡共置 + qwen3-asr 占太大 | `QWEN3_ASR_GPU_MEM=0.3` 或挪到独占卡, 见 `deployment.md §4.2` |
| 客户端代码读 `/asr/health` 报 KeyError: 'memory_usage' | §3 提到的破坏性变化 | 移除该字段引用 (新架构下网关进程拿不到 GPU 显存信息) |
| 网关 fd 耗尽 / Too many open files | 高 QPS + 默认 ulimit 偏低 | docker-compose 已加 `ulimits.nofile=65535`, 重启容器即生效 |

更多见 [`deployment.md §11`](./deployment.md)。
