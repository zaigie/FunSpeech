# -*- coding: utf-8 -*-
"""FunASR 子服务入口 — 占位

实际实现见 Step 3 (HTTP /asr/file, /asr/punc) 与 Step 4 (WS /asr/stream/v1)。
"""

from fastapi import FastAPI

app = FastAPI(title="funspeech-funasr-service")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "implemented": False}


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        log_level="info",
    )
