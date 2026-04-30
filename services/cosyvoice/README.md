# cosyvoice 子服务

承载 CosyVoice2 / CosyVoice3 的 TTS 推理和音色管理。**vLLM 在进程内加速 LLM 组件**(`CosyVoice2(load_vllm=True)`),不是 `vllm serve` 端到端,因此本子服务仍要装 `cosyvoice` Python 包及其全部依赖。

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
| `TTS_LOAD_VLLM` | `true` | 是否启用 vLLM 加速 |

## 接口

详细契约见 [plan](../../.claude/plans/bright-roaming-hummingbird.md)。占位实现仅提供 `GET /health`,Step 7 实装。

## 子模块

Step 7 会:
1. 把 `app/services/tts/third_party/CosyVoice` 这个 submodule 从主项目移到 `services/cosyvoice/third_party/CosyVoice`
2. 在 `server.py` 中通过 `sys.path.insert` 引入,`from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice3`

## 多副本注意

`POST /voices/add` 会写 `frontend.spk2info` 与 `voice_registry.json`。多 cosyvoice 副本场景下,网关侧必须把音色 CRUD 请求 sticky 到固定一个 primary 副本(否则状态会分裂),其它副本只读。本版本默认单副本,多副本作为后续 issue。
