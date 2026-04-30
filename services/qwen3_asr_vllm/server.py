# -*- coding: utf-8 -*-
"""Qwen3-ASR vLLM 子服务入口 — 占位

实际实现见 Step 6:
- HTTP /asr/file 走离线推理
- WS /asr/stream/v1 包装官方 streaming_transcribe + init_streaming_state
"""

from fastapi import FastAPI

app = FastAPI(title="funspeech-qwen3-asr-service")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "implemented": False}


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8003")),
        log_level="info",
    )
