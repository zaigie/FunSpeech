# services/

FunSpeech 微服务子服务集合。每个子目录是一个**独立 Python 项目**(独立 `pyproject.toml` + `uv.lock` + venv + Dockerfile),通过 HTTP / WebSocket 暴露推理能力。主项目(网关)通过 HTTP 调用,不再 import 模型代码。

| 子服务 | 端口 | 模型 | GPU | 说明 |
|---|---|---|---|---|
| `funasr` | 8001 | Paraformer / SenseVoice + VAD/PUNC | 是 | 离线 + 实时 WS |
| `dolphin` | 8002 | DataoceanAI Dolphin | 是 | 仅离线 |
| `qwen3_asr_vllm` | 8003 | Qwen3-ASR-1.7B | 是 (vLLM) | 离线 + 实时 WS |
| `cosyvoice` | 8004 | CosyVoice2 / CosyVoice3 | 是 (vLLM 加速 LLM) | TTS 离线 + 流式 + 音色管理 |
| `qwen3_tts` | 8005 | Qwen3-TTS Base | 是 | 开源本地 TTS 离线 + WS + 音色管理 |

依赖隔离的原因:Qwen3-ASR 要求 vLLM 0.11+ / transformers 4.57,与 funasr 1.2.6 + transformers 4.51.3 不兼容;CosyVoice 与 FunASR 的 PyTorch / CUDA 版本演进路线也独立。

## 启动

通过项目根的 `docker-compose.yml` 编排;或开发期单独起:

```bash
cd services/funasr
uv sync
uv run python server.py
```

## 内部接口约定

- 所有子服务 `GET /health` 返回 `{status: "healthy"|"unhealthy", ...}`
- 内部调用必须带 `X-Internal-Token: $INTERNAL_SERVICE_TOKEN` 头
- 详细接口契约见 [`/.claude/plans/bright-roaming-hummingbird.md`](../.claude/plans/bright-roaming-hummingbird.md) 中的"服务契约"小节
