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

## 二、构建期 HTTP 代理

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
docker compose up -d              # 默认: gateway + funasr + cosyvoice

docker compose --profile dolphin up -d                              # 加 dolphin
docker compose --profile qwen3-asr up -d                            # 加 qwen3-asr-vllm
docker compose --profile dolphin --profile qwen3-asr up -d          # 全部启动
```

## 四、服务编排与 GPU 拓扑

### 4.1 服务列表

| 服务 | 端口 | GPU | 默认启动 | profile |
|---|---|---|---|---|
| `gateway` | 8000 | ❌ | ✅ | (默认) |
| `funasr-0` | 8001 | ✅ | ✅ | (默认) |
| `dolphin-0` | 8002 | ✅ | ❌ | `dolphin` |
| `qwen3-asr-0` | 8003 | ✅ | ❌ | `qwen3-asr` |
| `cosyvoice-0` | 8004 | ✅ | ✅ | (默认) |

每个 GPU 子服务通过 `deploy.resources.reservations.devices` 申明 nvidia 设备,容器内通过 `CUDA_VISIBLE_DEVICES` 选卡。

### 4.2 单卡共置 vs 独占

进程内 vLLM 加速 ≠ vLLM serve。CosyVoice 与 Qwen3-ASR 都是"用 vLLM 跑模型里的 LLM 那一段",前后还有自己的 frontend / flow / vocoder / encoder / projector,**整体并发上限 = 副本数 = GPU 卡数**,不像通用 LLM 那样靠 continuous batching 把单卡吃满。

显存预算粗估(BF16/FP16,**不含 vLLM KV cache 池**):

| 子服务 | 模型权重 | 备注 |
|---|---|---|
| `funasr-0` | ~2.5–3 GB | Paraformer-Large + Paraformer-Online + VAD + PUNC × 2 |
| `dolphin-0` | ~0.6 GB | Dolphin Small |
| `qwen3-asr-0` | 模型 ~3.5–4 GB **+** KV cache 池 = `QWEN3_ASR_GPU_MEM × 卡显存` | 详见 §6.3 |
| `cosyvoice-0` | ~3.5 GB(`TTS_LOAD_VLLM=false` 时) | 启用 vLLM 再加 ~1 GB |

**典型组合:**

| 场景 | 卡 | 配置 |
|---|---|---|
| 8 GB 单卡 | 卡 0 | funasr + cosyvoice (vLLM=off),不开 qwen3-asr |
| 24 GB 单卡 | 卡 0 | funasr + cosyvoice + dolphin + qwen3-asr (`QWEN3_ASR_GPU_MEM=0.3`) |
| 24 GB 双卡 | 卡 0 | funasr + cosyvoice + dolphin |
| | 卡 1 | qwen3-asr 独占(`QWEN3_ASR_GPU_MEM=0.85`,吃满 KV cache 才能体现 vLLM 价值) |

### 4.3 配卡的两种方式

**方式 A — 全部贴卡 0(默认):** docker-compose.yml 里所有 GPU 子服务的 `CUDA_VISIBLE_DEVICES: "0"`,适合单卡或想让所有模型共置。

**方式 B — 跨卡分布:** 在 `.env` 或直接编辑 docker-compose.yml,把不同子服务指向不同卡:

```bash
# 把 qwen3-asr 移到卡 1, 其它留在卡 0
# 编辑 docker-compose.yml:
#   qwen3-asr-0:
#     environment:
#       CUDA_VISIBLE_DEVICES: "1"
```

注意 `device_ids: ["0", "1"]` 不会在子服务进程内启用张量并行(那需要 vLLM 的 `tensor_parallel_size`,且只对装不下的大模型有意义,本项目模型都是 0.5B–1.7B,装得下,不需要切)。多卡的正确用法是**多副本**(§5)。

## 五、多副本

### 5.1 添加副本

每张额外的卡 = 一个新副本。两步:

1. 在 `docker-compose.yml` 加新服务条目(`funasr-1`),把 `CUDA_VISIBLE_DEVICES` 设成另一张卡:
   ```yaml
   funasr-1:
     <<: *funasr-base   # 假设抽 anchor; 或直接复制 funasr-0 的定义
     container_name: funspeech-funasr-1
     environment:
       CUDA_VISIBLE_DEVICES: "1"
   ```
2. 网关 env 里把对应 URL 列表逗号分隔:
   ```
   FUNASR_SERVICE_URLS=http://funasr-0:8001,http://funasr-1:8001
   ```

网关侧 `_HttpReplicaPool` 会做最少连接 + 随机选副本调度;每个外部 WS 会话**绑定固定副本**(session 级亲和性),保证 cache 状态不串。

### 5.2 cosyvoice 多副本的写副本限制

**音色 CRUD(`POST/DELETE /voices/*`)必须打到固定 primary 副本**,否则 `spk2info.pt` 写操作会冲突。当前网关无 sticky 路由,生产多副本部署需自行加 LB 规则,或只让一个副本接收写请求(其它副本只读,定期重启同步 spk2info)。

## 六、子服务环境变量参考

所有 GPU 子服务共有的:

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | 子服务各自约定(8001/2/3/4) | 监听端口 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `INTERNAL_SERVICE_TOKEN` | `funspeech-internal` | 网关→子服务鉴权头(`X-Internal-Token`) |
| `MODELSCOPE_PATH` | `~/.cache/modelscope/hub` | 容器内权重缓存路径(已通过 bind mount 共享) |
| `CUDA_VISIBLE_DEVICES` | `0` | 选卡;改成 `1` / `0,1` / 空字符串(强制 CPU)等 |

下面分子服务列出独有的 env。

### 6.1 funasr-0 (端口 8001)

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

显存:`offline` 模式只占 ~0.7 GB;`all` 模式约 2.5–3 GB(两个 paraformer + VAD + 双 PUNC)。

### 6.2 dolphin-0 (端口 8002,profile=dolphin)

| 变量 | 默认 | 说明 |
|---|---|---|
| `DOLPHIN_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0`(注意 dolphin 用 `cuda` 不带索引,索引由 `CUDA_VISIBLE_DEVICES` 控制) |
| `DOLPHIN_SIZE` | `small` | dolphin 模型规模(目前仅 `small`) |
| `DOLPHIN_MODEL_PATH` | `DataoceanAI/dolphin-small` | 模型 id (相对 `MODELSCOPE_PATH`) |

显存约 0.6 GB,可与 funasr 同卡共置。

### 6.3 qwen3-asr-0 (端口 8003,profile=qwen3-asr)

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWEN3_ASR_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | vLLM 加载的模型 id |
| `QWEN3_ASR_GPU_MEM` | `0.8` | **vLLM 启动时直接预留这么大比例的卡显存做 KV cache 池** |
| `QWEN3_ASR_MAX_NEW_TOKENS` | `32` | 单步生成上限,影响延迟与显存 |
| `QWEN3_UNFIXED_CHUNK_NUM` | `2` | 流式状态机 unfixed chunk 数(token 修订窗口) |
| `QWEN3_UNFIXED_TOKEN_NUM` | `5` | 流式状态机 unfixed token 数 |
| `QWEN3_CHUNK_SIZE_SEC` | `2.0` | 流式 chunk 时长(秒) |

**`QWEN3_ASR_GPU_MEM` 是显存调度最关键的 env:**

| 卡显存 | 与 funasr+cosyvoice 同卡共置 | 独占整张卡 |
|---|---|---|
| 8 GB | 不建议(权重 + 共置就快爆) | `0.85`(显存 ≈ 6.8 GB,够 1.7B 模型 + 小 KV) |
| 16 GB | `0.4`(预留 ~9.6 GB 给别的) | `0.85` |
| 24 GB | `0.3`(预留 ~16 GB 给别的) | `0.85`–`0.9` |
| 40+ GB | `0.2`–`0.3` 即可 | `0.85` |

`gpu_memory_utilization` 越大 KV cache 池越大,vLLM 能并行处理的请求越多;但子服务进程内并发仍受 Python GIL + 单进程模型限制,实际收益要看负载。

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

**显存:**

- `clone` only(默认 CosyVoice3):~3.5 GB
- `sft` only:~1.5 GB
- `all`(双模型同时加载):~5 GB
- 启用 TRT:加 ~0.5–1 GB
- 启用 vLLM:加 ~1 GB(LLM 段被 vLLM 接管,有自己的 KV cache)

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
| `FUNASR_SERVICE_URLS` | `http://funasr-0:8001` | 逗号分隔多副本 |
| `DOLPHIN_SERVICE_URLS` | `http://dolphin-0:8002` | |
| `QWEN3_ASR_SERVICE_URLS` | `http://qwen3-asr-0:8003` | |
| `COSYVOICE_SERVICE_URLS` | `http://cosyvoice-0:8004` | |
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
| `${MODELSCOPE_CACHE}` (默认 `~/.cache/modelscope`) | `/root/.cache/modelscope` | 模型权重缓存,所有 GPU 子服务共享 | 提前下载可省首次启动等待 |
| `./voices` | `/app/voices`(只在 cosyvoice) | 零样本克隆音色 + spk2info.pt | 持久化用户数据 |
| `./temp` | `/app/temp`(只在 gateway) | 网关临时音频文件 | |
| `./data` | `/app/data`(只在 gateway) | 异步 TTS 任务库 | |
| `./logs` | `/app/logs`(只在 gateway) | 日志 | |

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
docker compose exec qwen3-asr-0 nvidia-smi   # 看容器视角(受 CUDA_VISIBLE_DEVICES 限制)

# 重启某个服务(不影响其它)
docker compose restart funasr-0

# 完整清理(注意会丢容器,保留卷)
docker compose down

# 连卷一起清(会丢音色数据!)
docker compose down -v
```

### 常见问题

- **`out of memory` / `CUDA OOM`**:多半是 qwen3-asr 与别的服务同卡共置但 `QWEN3_ASR_GPU_MEM` 太大,降到 `0.3`–`0.4` 再试。
- **网关一直报 503,但子服务日志看着正常**:看 `docker compose ps` 是否 `(healthy)`;若 healthcheck 还在 `start_period`,等加载完。
- **funasr 子服务挂了**:旧版本 `services/funasr/server.py` 缺 import,WS 流式会 NameError——升级到 `0783c3b` 之后的版本即可。
- **CosyVoice3 输出全是噪音**:`TTS_ENABLE_FP16=true` + `TTS_LOAD_TRT=true` 同时开会有 NaN,关一个。
