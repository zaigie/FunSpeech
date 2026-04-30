# -*- coding: utf-8 -*-
"""CosyVoice 子服务入口 — 占位

实际实现见 Step 7:
- POST /tts/file
- WS  /tts/stream
- POST /voices/add, DELETE /voices/{name}, GET /voices
- POST /text/normalize
"""

from fastapi import FastAPI

app = FastAPI(title="funspeech-cosyvoice-service")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "implemented": False}


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8004")),
        log_level="info",
    )
