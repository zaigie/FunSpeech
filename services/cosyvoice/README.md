# cosyvoice 子服务

承载 CosyVoice2 / CosyVoice3 的 TTS 推理和音色管理。**vLLM 在进程内加速 LLM 组件**(`CosyVoice2(load_vllm=True)`),不是 `vllm serve` 端到端,因此本子服务仍要装 `cosyvoice` Python 包及其全部依赖。

> **🚨 Build 前必须做**: 上游 `CosyVoice` 源码以 git submodule 形式引入到 `services/cosyvoice/third_party/CosyVoice`。在宿主机执行
>
> ```bash
> git submodule update --init --recursive
> ```
>
> 之后再 `docker compose build`。否则镜像里 `third_party/CosyVoice` 是空目录,容器启动时 `from cosyvoice.cli.cosyvoice import ...` 立刻 `ModuleNotFoundError`。

## 启动

```bash
uv sync
uv run python server.py
# 默认端口 8004
```

环境变量:

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `8004` | 监听端口 |
| `TTS_MODEL_MODE` | `all` | `all` / `sft` / `clone` |
| `CLONE_MODEL_VERSION` | `cosyvoice3` | `cosyvoice2` / `cosyvoice3` |
| `COSYVOICE3_MODEL_ID` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 模型 ID |
| `MODELSCOPE_PATH` | `~/.cache/modelscope/hub` | 模型权重缓存目录 |
| `VOICES_DIR` | `/app/voices` | 音色注册表与 spk2info.pt 持久化目录(由网关 mount) |
| `INTERNAL_SERVICE_TOKEN` |  | 网关调用必须携带 `X-Internal-Token` |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU 绑定(每副本一卡) |
| `TTS_LOAD_TRT` | `false` | TensorRT 加速 (CosyVoice3 + FP16 + TRT 有 NaN 风险, FP16 建议关) |
| `TTS_ENABLE_FP16` | `false` | FP16 推理 |
| `TTS_LOAD_VLLM` | `false` | 进程内 vLLM 加速 LLM 阶段 (与本子服务的 `transformers==4.51.3` 冲突, 启用前请确认 — 见下) |

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查,返回 `sft_loaded` / `clone_loaded` / `clone_model_version` 等 |
| POST | `/tts/file` | 整段离线合成,JSON body,响应 WAV bytes + `X-Native-Sample-Rate` / `X-Sentences` header |
| WS | `/tts/stream` | 流式合成,首帧 JSON 起协,后续二进制 float32 PCM,末帧 `{"type":"done"}` |
| GET | `/voices` | 列出 preset / clone / all 三类音色 + registry |
| GET | `/voices/{name}` | 单条音色信息 |
| POST | `/voices` | 注册新克隆音色 (multipart: `name`, `prompt_text`, `audio`) |
| DELETE | `/voices/{name}` | 删除音色 |
| POST | `/voices/refresh` | 扫描 `VOICES_DIR/*.txt` + `*.wav` 自动注册 |
| POST | `/voices/reload` | 从磁盘热重载 spk2info + registry (多副本同步用) |
| POST | `/text/normalize` | 切句 |

## vLLM 兼容性注意

vLLM 加速 CosyVoice2 LLM 需要 `transformers>=4.55`,与本子服务的 `transformers==4.51.3` 不兼容。如需启用 `TTS_LOAD_VLLM=true`,要么:

1. 单独建一个 vLLM 子服务,本子服务 `TTS_LOAD_VLLM=false`;或
2. 在本 venv 外手动 `pip install --no-deps vllm` 自担依赖冲突风险。

上游 `README.md` 也建议 vllm 用独立 conda env。

## 多副本部署

`POST /voices` / `DELETE /voices/{name}` 会写 `frontend.spk2info` 与 `voice_registry.json`。多 cosyvoice 副本场景下:

1. **写副本约定**: `COSYVOICE_SERVICE_URLS` 列表中的**第一个 URL** = 主写副本,所有音色 CRUD 都打到它
2. **读副本同步**: 网关在写完成后会向其它副本广播 `POST /voices/reload`,让它们从磁盘热重载 spk2info + registry
3. 默认 `docker-compose.yml` 是单副本,多副本时需要:
   - `docker compose up -d --scale cosyvoice-0=N` (但要先把 `gpu device_ids` 改成 `deploy.resources` 形式)
   - 在网关 env 里把 `COSYVOICE_SERVICE_URLS` 改成 `http://cosyvoice-0:8004,http://cosyvoice-1:8004,...`
