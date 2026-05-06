# -*- coding: utf-8 -*-
"""Qwen3-ASR + vLLM 子服务

vLLM 在进程内加速 Qwen2.5-1.5B 等 LLM backbone。流式调用走官方
qwen_asr.Qwen3ASRModel 的 init_streaming_state / streaming_transcribe /
finish_streaming_transcribe API,具备跨段上下文与 token 修订能力,
比 vLLM 通用 /v1/realtime 端点质量更好(参见 vllm Issue #35767)。

接口:
    GET  /health
    POST /asr/file        (multipart 离线整段)
    WS   /asr/stream/v1   (流式, 协议与 funasr 一致以便网关复用)

启动:
    PORT=8003 uv run python server.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)


logger = logging.getLogger("qwen3_asr_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("QWEN3_ASR_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
GPU_MEMORY_UTILIZATION = float(os.getenv("QWEN3_ASR_GPU_MEM", "0.8"))
# 官方推荐 max_new_tokens=4096, max_inference_batch_size=128
# 见 https://github.com/QwenLM/Qwen3-ASR
MAX_NEW_TOKENS = int(os.getenv("QWEN3_ASR_MAX_NEW_TOKENS", "4096"))
MAX_INFERENCE_BATCH_SIZE = int(os.getenv("QWEN3_ASR_MAX_BATCH", "128"))
SAMPLE_RATE = 16000  # qwen3-asr 固定 16kHz
INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")

# 流式参数
DEFAULT_UNFIXED_CHUNK_NUM = int(os.getenv("QWEN3_UNFIXED_CHUNK_NUM", "2"))
DEFAULT_UNFIXED_TOKEN_NUM = int(os.getenv("QWEN3_UNFIXED_TOKEN_NUM", "5"))
DEFAULT_CHUNK_SIZE_SEC = float(os.getenv("QWEN3_CHUNK_SIZE_SEC", "2.0"))


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

_model = None


def _load_model():
    """初始化 Qwen3ASRModel (vLLM backend)"""
    global _model
    if _model is not None:
        return _model

    from qwen_asr import Qwen3ASRModel

    logger.info(
        "加载 Qwen3-ASR 模型: %s (gpu_mem=%.2f, max_tokens=%d)",
        MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_NEW_TOKENS,
    )
    _model = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_new_tokens=MAX_NEW_TOKENS,
        max_inference_batch_size=MAX_INFERENCE_BATCH_SIZE,
    )
    return _model


# ---------------------------------------------------------------------------
# 业务逻辑
# ---------------------------------------------------------------------------


def _decode_audio_to_16k_mono(audio_bytes: bytes, declared_format: str = "wav") -> np.ndarray:
    """把任意支持的容器/采样率解成 16k float32 单声道"""
    audio, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # 简易重采样(线性插值)。生产可改 librosa/torchaudio。
        ratio = SAMPLE_RATE / sr
        new_len = int(round(len(audio) * ratio))
        if new_len > 0:
            x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
            x_new = np.linspace(0, 1, num=new_len, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


def _decode_pcm_int16(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_file_offline(
    audio_bytes: bytes, audio_format: str, language: Optional[str] = None
) -> Dict[str, Any]:
    """离线整段识别。复用流式 API 一次性跑完。"""
    model = _load_model()
    wav = _decode_audio_to_16k_mono(audio_bytes, declared_format=audio_format)

    state = model.init_streaming_state(
        unfixed_chunk_num=DEFAULT_UNFIXED_CHUNK_NUM,
        unfixed_token_num=DEFAULT_UNFIXED_TOKEN_NUM,
        chunk_size_sec=DEFAULT_CHUNK_SIZE_SEC,
    )

    # 一次性整段送进去
    model.streaming_transcribe(wav, state)
    model.finish_streaming_transcribe(state)
    return {
        "text": getattr(state, "text", "") or "",
        "language": getattr(state, "language", "") or "",
    }


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
        logger.info("Qwen3-ASR 子服务就绪")
    except Exception as exc:
        logger.error("Qwen3-ASR 模型预加载失败: %s", exc, exc_info=True)
    yield


app = FastAPI(title="funspeech-qwen3-asr-service", lifespan=lifespan)


def _check_internal_token(request: Request) -> None:
    if not INTERNAL_SERVICE_TOKEN:
        return
    token = request.headers.get("X-Internal-Token", "")
    if token != INTERNAL_SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="invalid internal token")


def _ws_check_token(websocket: WebSocket) -> bool:
    if not INTERNAL_SERVICE_TOKEN:
        return True
    token = websocket.query_params.get("token") or ""
    return token == INTERNAL_SERVICE_TOKEN


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "model": MODEL_PATH,
        "model_loaded": _model is not None,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/asr/file")
async def asr_file(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form(""),
    sample_rate: int = Form(16000),
) -> dict:
    _check_internal_token(request)

    audio_bytes = await audio.read()
    audio_format = (
        os.path.splitext(audio.filename or "audio.wav")[1].lstrip(".").lower()
        or "wav"
    )

    try:
        result = transcribe_file_offline(
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            language=language or None,
        )
    except Exception as exc:
        logger.exception("识别失败")
        raise HTTPException(status_code=500, detail=f"transcribe error: {exc}")

    return {**result, "sample_rate": SAMPLE_RATE}


@app.websocket("/asr/stream/v1")
async def asr_stream(websocket: WebSocket) -> None:
    """流式协议与 funasr 子服务保持一致,以便网关侧统一处理"""
    if not _ws_check_token(websocket):
        await websocket.close(code=4401, reason="invalid internal token")
        return

    await websocket.accept()

    started = False
    state = None
    audio_format = "pcm"
    last_text = ""
    last_language = ""
    unfixed_chunk_num = DEFAULT_UNFIXED_CHUNK_NUM
    unfixed_token_num = DEFAULT_UNFIXED_TOKEN_NUM
    chunk_size_sec = DEFAULT_CHUNK_SIZE_SEC

    try:
        while True:
            msg = await websocket.receive()

            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "invalid json"})
                    continue

                op = payload.get("op")
                if op == "start":
                    audio_format = (payload.get("format") or "pcm").lower()
                    unfixed_chunk_num = int(
                        payload.get("unfixed_chunk_num", DEFAULT_UNFIXED_CHUNK_NUM)
                    )
                    unfixed_token_num = int(
                        payload.get("unfixed_token_num", DEFAULT_UNFIXED_TOKEN_NUM)
                    )
                    chunk_size_sec = float(
                        payload.get("chunk_size_sec", DEFAULT_CHUNK_SIZE_SEC)
                    )

                    model = _load_model()
                    state = model.init_streaming_state(
                        unfixed_chunk_num=unfixed_chunk_num,
                        unfixed_token_num=unfixed_token_num,
                        chunk_size_sec=chunk_size_sec,
                    )
                    started = True
                    last_text = ""
                    last_language = ""
                    await websocket.send_json({"type": "started"})

                elif op == "flush":
                    if not started or state is None:
                        await websocket.send_json({"type": "error", "message": "not started"})
                        continue
                    model = _load_model()
                    model.finish_streaming_transcribe(state)
                    final_text = getattr(state, "text", "") or ""
                    await websocket.send_json({"type": "flushed", "text": final_text})
                    # 重置 state, 下一句重新 init
                    state = None
                    started = False

                elif op == "close":
                    break

                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"unknown op: {op}"}
                    )

            elif "bytes" in msg:
                if not started or state is None:
                    await websocket.send_json({"type": "error", "message": "not started"})
                    continue

                audio_bytes = msg["bytes"]
                try:
                    if audio_format == "pcm":
                        audio_array = _decode_pcm_int16(audio_bytes)
                    elif audio_format in ("wav", "wave"):
                        audio_array = _decode_audio_to_16k_mono(audio_bytes, audio_format)
                    else:
                        await websocket.send_json(
                            {"type": "error", "message": f"unsupported format: {audio_format}"}
                        )
                        continue
                except Exception as exc:
                    await websocket.send_json(
                        {"type": "error", "message": f"audio decode: {exc}"}
                    )
                    continue

                is_silence = bool(
                    len(audio_array) > 0
                    and float(np.max(np.abs(audio_array))) < 0.002
                )

                try:
                    model = _load_model()
                    model.streaming_transcribe(audio_array, state)
                    cur_text = getattr(state, "text", "") or ""
                    cur_lang = getattr(state, "language", "") or ""
                except Exception as exc:
                    logger.exception("streaming chunk failed")
                    await websocket.send_json(
                        {"type": "error", "message": f"asr error: {exc}"}
                    )
                    continue

                # 增量文本(本次新增部分),便于网关侧累积展示
                if cur_text.startswith(last_text):
                    delta = cur_text[len(last_text):]
                else:
                    # state.text 可能因 token revision 而部分回退
                    delta = cur_text
                last_text = cur_text
                last_language = cur_lang

                await websocket.send_json(
                    {
                        "type": "partial",
                        # 与 funasr 协议保持一致字段名
                        "text": cur_text,
                        "text_punc": "",
                        "is_silence": is_silence,
                        # qwen3 特有补充字段
                        "delta": delta,
                        "language": cur_lang,
                    }
                )

    except WebSocketDisconnect:
        logger.info("WS 客户端断开")
    except Exception:
        logger.exception("WS 处理异常")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8003")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
