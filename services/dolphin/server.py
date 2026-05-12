# -*- coding: utf-8 -*-
"""Dolphin ASR 子服务

仅离线 HTTP /asr/file (DataoceanAI Dolphin Small)。
实时流式不支持(模型限制)。

启动:
    PORT=8002 uv run python server.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse


logger = logging.getLogger("dolphin_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

DOLPHIN_DEVICE = os.getenv("DOLPHIN_DEVICE", "auto")
DOLPHIN_SIZE = os.getenv("DOLPHIN_SIZE", "small")
DOLPHIN_MODEL_PATH = os.getenv("DOLPHIN_MODEL_PATH", "DataoceanAI/dolphin-small")
MODELSCOPE_PATH = os.path.expanduser(
    os.getenv("MODELSCOPE_PATH", "~/.cache/modelscope/hub")
)
INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")
# GPU 串行进入 (同 funasr 理由), 横向扩展用多副本
GPU_INFERENCE_CONCURRENCY = 1


def _detect_device(device_hint: str) -> str:
    """与主项目 DolphinEngine 的设备探测保持一致 — Dolphin 用 'cuda' 而非 'cuda:0'"""
    if device_hint and device_hint not in ("auto", ""):
        if device_hint.startswith("cuda:"):
            return "cuda"
        return device_hint

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = _detect_device(DOLPHIN_DEVICE)


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    import dolphin

    full_path = os.path.join(MODELSCOPE_PATH, DOLPHIN_MODEL_PATH)
    logger.info(
        "加载 Dolphin 模型: size=%s path=%s device=%s", DOLPHIN_SIZE, full_path, DEVICE
    )
    _model = dolphin.load_model(DOLPHIN_SIZE, full_path, DEVICE)
    return _model


# ---------------------------------------------------------------------------
# 业务逻辑
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]*>")
_TIMESTAMP_RE = re.compile(r"<[\d.]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub("", text)
    text = _TIMESTAMP_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def transcribe(
    audio_path: str,
    lang_sym: str = "zh",
    region_sym: str = "SHANGHAI",
) -> str:
    import dolphin

    model = _load_model()
    waveform = dolphin.load_audio(audio_path)

    if lang_sym and region_sym:
        result = model(waveform, lang_sym=lang_sym, region_sym=region_sym)
    else:
        result = model(waveform)

    raw = result.text if hasattr(result, "text") else str(result)
    return _clean_text(raw)


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------


_gpu_semaphore: Optional[asyncio.Semaphore] = None
_load_failed: bool = False
_load_error_msg: str = ""


def _get_gpu_semaphore() -> asyncio.Semaphore:
    global _gpu_semaphore
    if _gpu_semaphore is None:
        _gpu_semaphore = asyncio.Semaphore(GPU_INFERENCE_CONCURRENCY)
    return _gpu_semaphore


async def _run_inference(func, *args, **kwargs):
    sem = _get_gpu_semaphore()
    async with sem:
        return await asyncio.to_thread(func, *args, **kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _load_failed, _load_error_msg
    try:
        _load_model()
        _get_gpu_semaphore()
        logger.info("Dolphin 子服务就绪 (device=%s)", DEVICE)
    except Exception as exc:
        _load_failed = True
        _load_error_msg = str(exc)
        logger.error("Dolphin 模型预加载失败: %s", exc, exc_info=True)
    yield


app = FastAPI(title="funspeech-dolphin-service", lifespan=lifespan)


def _check_internal_token(request: Request) -> None:
    if not INTERNAL_SERVICE_TOKEN:
        return
    token = request.headers.get("X-Internal-Token", "")
    if token != INTERNAL_SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="invalid internal token")


@app.get("/health")
async def health() -> dict:
    body = {
        "status": "healthy",
        "device": DEVICE,
        "size": DOLPHIN_SIZE,
        "model_loaded": _model is not None,
    }
    if _load_failed:
        body["status"] = "unhealthy"
        body["error"] = _load_error_msg
        return JSONResponse(status_code=503, content=body)
    if _model is None:
        body["status"] = "starting"
        return JSONResponse(status_code=503, content=body)
    return body


@app.post("/asr/file")
async def asr_file(
    request: Request,
    audio: UploadFile = File(...),
    lang_sym: str = Form("zh"),
    region_sym: str = Form("SHANGHAI"),
    sample_rate: int = Form(16000),
) -> dict:
    _check_internal_token(request)

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        text = await _run_inference(
            transcribe,
            audio_path=tmp_path,
            lang_sym=lang_sym,
            region_sym=region_sym,
        )
    except Exception as exc:
        logger.exception("识别失败")
        raise HTTPException(status_code=500, detail=f"transcribe error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # 注: 标点恢复与 ITN 不在 dolphin 子服务做; 网关侧若需要,
    # 应调 funasr 子服务的 /asr/punc 后再 ITN
    return {"text": text, "sample_rate": sample_rate}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8002")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
