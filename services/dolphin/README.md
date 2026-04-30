# dolphin 子服务

承载 DataoceanAI Dolphin Small 模型,仅离线 HTTP 识别。

## 启动

```bash
uv sync
uv run python server.py
# 默认端口 8002
```

环境变量:

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | `8002` | 监听端口 |
| `MODELSCOPE_PATH` | `~/.cache/modelscope/hub` | 模型权重缓存目录 |
| `INTERNAL_SERVICE_TOKEN` |  | 网关调用必须携带 `X-Internal-Token` |

## 接口

详细契约见 [plan](../../.claude/plans/bright-roaming-hummingbird.md)。占位实现仅提供 `GET /health`,Step 5 实装 `POST /asr/file`。
