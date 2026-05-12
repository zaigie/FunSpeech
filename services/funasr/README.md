# funasr 子服务

承载 FunASR 系列模型(Paraformer Large、SenseVoice Small)、VAD、标点恢复模型。对外提供 HTTP 离线识别和 WebSocket 流式识别。

## 启动

```bash
uv sync
uv run python server.py
# 默认端口 8001
```

环境变量:

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `8001` | 监听端口 |
| `ASR_MODEL_MODE` | `all` | `all` / `offline` / `realtime` |
| `MODELSCOPE_PATH` | `~/.cache/modelscope/hub` | 模型权重缓存目录(与主项目一致) |
| `INTERNAL_SERVICE_TOKEN` |  | 网关调用本服务时必须携带 `X-Internal-Token` |

## 接口

详细契约见 [plan](../../.claude/plans/bright-roaming-hummingbird.md)。占位实现仅提供 `GET /health`,其余在 Step 3 / Step 4 实装。
