# -*- coding: utf-8 -*-
"""Dolphin 子服务入口 — 占位

实际实现见 Step 5 (HTTP /asr/file)。
"""

from fastapi import FastAPI

app = FastAPI(title="funspeech-dolphin-service")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "implemented": False}


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8002")),
        log_level="info",
    )
