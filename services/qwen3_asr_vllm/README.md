# qwen3_asr_vllm 子服务

承载阿里 Qwen3-ASR 系列模型(默认 `Qwen/Qwen3-ASR-1.7B`)。**vLLM 在进程内加速 LLM**,而不是走 `vllm serve` 端到端。这样我们可以利用官方 [`example_qwen3_asr_vllm_streaming.py`](https://github.com/QwenLM/Qwen3-ASR/blob/main/examples/example_qwen3_asr_vllm_streaming.py) 提供的 `init_streaming_state(unfixed_chunk_num, unfixed_token_num)` + `streaming_transcribe()` 接口,具备跨段上下文与 token 修订能力。vLLM 自带的 `/v1/realtime` 端点对 Qwen3-ASR 质量明显劣化(参考 vllm Issue #35767),不在本项目使用。

## 启动

```bash
uv sync
uv run python server.py
# 默认端口 8003
```

环境变量:

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `8003` | 监听端口 |
| `QWEN3_ASR_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace / ModelScope 模型 ID;离线部署时设为本地路径,如 `/root/.cache/modelscope/hub/Qwen/Qwen3-ASR-1.7B` |
| `INTERNAL_SERVICE_TOKEN` |  | 网关调用必须携带 `X-Internal-Token` |
| `CUDA_VISIBLE_DEVICES` | `0` | 容器内 GPU 序号;如果 compose 已用 `device_ids` 透传单张宿主卡,这里保持 `0` |

## 接口

详细契约见 [plan](../../.claude/plans/bright-roaming-hummingbird.md)。占位实现仅提供 `GET /health`,Step 6 实装离线与流式。

## 子模块

Step 6 会在本目录下加 git submodule `third_party/Qwen3-ASR` 指向 `https://github.com/QwenLM/Qwen3-ASR`,以使用官方 `Qwen3ASRModel` 与流式状态机。
