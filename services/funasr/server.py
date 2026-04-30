# -*- coding: utf-8 -*-
"""FunASR 子服务

暴露:
- GET  /health
- POST /asr/file          (multipart 音频文件)
- POST /asr/punc          (仅打标点)
- WS   /asr/stream/v1     (实时流式, Step 4 实装)

启动:
    PORT=8001 uv run python server.py
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional

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


logger = logging.getLogger("funasr_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

ASR_MODEL_MODE = os.getenv("ASR_MODEL_MODE", "all").lower()
ASR_DEVICE = os.getenv("ASR_DEVICE", "auto")
MODELSCOPE_PATH = os.path.expanduser(
    os.getenv("MODELSCOPE_PATH", "~/.cache/modelscope/hub")
)
INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")

OFFLINE_MODEL = os.getenv(
    "FUNASR_OFFLINE_MODEL",
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)
REALTIME_MODEL = os.getenv(
    "FUNASR_REALTIME_MODEL",
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
)
VAD_MODEL = os.getenv("VAD_MODEL", "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
VAD_MODEL_REVISION = os.getenv("VAD_MODEL_REVISION", "v2.0.4")
PUNC_MODEL = os.getenv(
    "PUNC_MODEL", "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
)
PUNC_MODEL_REVISION = os.getenv("PUNC_MODEL_REVISION", "v2.0.4")
PUNC_REALTIME_MODEL = os.getenv(
    "PUNC_REALTIME_MODEL",
    "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
)

# 全局 funasr kwargs(与主项目当前一致)
FUNASR_AUTOMODEL_KWARGS = {
    "trust_remote_code": False,
    "disable_update": True,
    "disable_pbar": True,
    "disable_log": True,
}


# ---------------------------------------------------------------------------
# 设备探测
# ---------------------------------------------------------------------------


def _detect_device(device_hint: str) -> str:
    if device_hint and device_hint not in ("auto", ""):
        return device_hint

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


DEVICE = _detect_device(ASR_DEVICE)


# ---------------------------------------------------------------------------
# 模型缓存(VAD / PUNC / 实时 PUNC 按需懒加载,与主项目原行为一致)
# ---------------------------------------------------------------------------


_offline_model = None
_realtime_model = None

_vad_model: Optional[object] = None
_punc_model: Optional[object] = None
_punc_realtime_model: Optional[object] = None
_aux_lock = threading.Lock()


def _load_offline_model():
    global _offline_model
    if _offline_model is not None:
        return _offline_model
    from funasr import AutoModel

    logger.info("加载离线 FunASR 模型: %s (device=%s)", OFFLINE_MODEL, DEVICE)
    _offline_model = AutoModel(
        model=OFFLINE_MODEL, device=DEVICE, **FUNASR_AUTOMODEL_KWARGS
    )
    return _offline_model


def _load_realtime_model():
    global _realtime_model
    if _realtime_model is not None:
        return _realtime_model
    from funasr import AutoModel

    logger.info("加载实时 FunASR 模型: %s (device=%s)", REALTIME_MODEL, DEVICE)
    _realtime_model = AutoModel(
        model=REALTIME_MODEL, device=DEVICE, **FUNASR_AUTOMODEL_KWARGS
    )
    return _realtime_model


def _get_vad_model():
    global _vad_model
    if _vad_model is not None:
        return _vad_model
    from funasr import AutoModel

    with _aux_lock:
        if _vad_model is None:
            logger.info("加载 VAD 模型: %s", VAD_MODEL)
            _vad_model = AutoModel(
                model=VAD_MODEL,
                model_revision=VAD_MODEL_REVISION,
                device=DEVICE,
                **FUNASR_AUTOMODEL_KWARGS,
            )
    return _vad_model


def _get_punc_model(realtime: bool = False):
    global _punc_model, _punc_realtime_model
    cache = "_punc_realtime_model" if realtime else "_punc_model"
    cached = globals()[cache]
    if cached is not None:
        return cached
    from funasr import AutoModel

    with _aux_lock:
        if globals()[cache] is None:
            model_id = PUNC_REALTIME_MODEL if realtime else PUNC_MODEL
            logger.info("加载 PUNC 模型: %s", model_id)
            instance = AutoModel(
                model=model_id,
                model_revision=PUNC_MODEL_REVISION,
                device=DEVICE,
                **FUNASR_AUTOMODEL_KWARGS,
            )
            globals()[cache] = instance
    return globals()[cache]


# ---------------------------------------------------------------------------
# ITN
# ---------------------------------------------------------------------------

_itn_normalizer = None


def _apply_itn(text: str) -> str:
    global _itn_normalizer
    if not text:
        return text
    try:
        if _itn_normalizer is None:
            from wetext import Normalizer

            _itn_normalizer = Normalizer(lang="zh", operator="itn")
        return _itn_normalizer.normalize(text)
    except Exception as exc:
        logger.warning("ITN 处理失败,原文返回: %s", exc)
        return text


# ---------------------------------------------------------------------------
# 业务逻辑(从主项目 FunASREngine 搬过来)
# ---------------------------------------------------------------------------


def transcribe_offline(
    audio_path: str,
    hotwords: str = "",
    enable_punctuation: bool = False,
    enable_itn: bool = False,
    enable_vad: bool = False,
) -> str:
    offline = _load_offline_model()

    if enable_vad:
        # 复用主项目的"临时 AutoModel + 注入 VAD/PUNC"技巧
        from funasr import AutoModel
        import types

        vad_inst = _get_vad_model()
        punc_inst = _get_punc_model() if enable_punctuation else None

        temp = type("TempAutoModel", (), {})()
        temp.model = offline.model
        temp.kwargs = offline.kwargs
        temp.model_path = offline.model_path
        temp.spk_model = None
        temp.vad_model = vad_inst.model
        temp.vad_kwargs = vad_inst.kwargs
        if punc_inst:
            temp.punc_model = punc_inst.model
            temp.punc_kwargs = punc_inst.kwargs
        else:
            temp.punc_model = None
            temp.punc_kwargs = {}
        temp.inference = types.MethodType(AutoModel.inference, temp)
        temp.inference_with_vad = types.MethodType(AutoModel.inference_with_vad, temp)
        temp.generate = types.MethodType(AutoModel.generate, temp)

        result = temp.generate(
            input=audio_path,
            hotword=hotwords or None,
            cache={},
        )
    else:
        result = offline.generate(
            input=audio_path,
            hotword=hotwords or None,
            cache={},
        )
        if enable_punctuation and result and len(result) > 0:
            text = (result[0].get("text") or "").strip()
            if text:
                punc_inst = _get_punc_model()
                punc_result = punc_inst.generate(input=text, cache={})
                if punc_result and len(punc_result) > 0:
                    result[0]["text"] = punc_result[0].get("text", text)

    if not result or not len(result):
        return ""
    text = (result[0].get("text") or "").strip()
    if enable_itn and text:
        text = _apply_itn(text)
    return text


def transcribe_realtime_chunk(
    audio_array,
    cache: dict,
    is_final: bool,
    chunk_size: list,
    encoder_chunk_look_back: int,
    decoder_chunk_look_back: int,
) -> str:
    """单步 funasr 实时流式推理。cache 按引用更新。"""
    realtime = _load_realtime_model()
    result = realtime.generate(
        input=audio_array,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back,
    )
    if not result or not len(result):
        return ""
    return (result[0].get("text") or "").strip()


def punc_realtime_apply(text: str, cache: dict) -> str:
    """实时标点。cache 按引用更新。"""
    if not text:
        return text
    punc = _get_punc_model(realtime=True)
    result = punc.generate(input=text, cache=cache)
    if not result or not len(result):
        return text
    return (result[0].get("text") or text).strip()


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时根据 mode 预加载,避免首次请求长等待
    try:
        if ASR_MODEL_MODE in ("all", "offline"):
            _load_offline_model()
        if ASR_MODEL_MODE in ("all", "realtime"):
            _load_realtime_model()
        logger.info("FunASR 子服务就绪 (mode=%s, device=%s)", ASR_MODEL_MODE, DEVICE)
    except Exception as exc:
        logger.error("模型预加载失败: %s", exc, exc_info=True)
    yield


app = FastAPI(title="funspeech-funasr-service", lifespan=lifespan)


def _check_internal_token(request: Request) -> None:
    if not INTERNAL_SERVICE_TOKEN:
        return
    token = request.headers.get("X-Internal-Token", "")
    if token != INTERNAL_SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="invalid internal token")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "device": DEVICE,
        "mode": ASR_MODEL_MODE,
        "offline_loaded": _offline_model is not None,
        "realtime_loaded": _realtime_model is not None,
        "vad_loaded": _vad_model is not None,
        "punc_loaded": _punc_model is not None,
        "punc_realtime_loaded": _punc_realtime_model is not None,
    }


@app.post("/asr/file")
async def asr_file(
    request: Request,
    audio: UploadFile = File(...),
    hotwords: str = Form(""),
    enable_punctuation: bool = Form(False),
    enable_itn: bool = Form(False),
    enable_vad: bool = Form(False),
    sample_rate: int = Form(16000),
) -> dict:
    _check_internal_token(request)

    if ASR_MODEL_MODE not in ("all", "offline"):
        raise HTTPException(
            status_code=503,
            detail=f"offline model not loaded in this service (mode={ASR_MODEL_MODE})",
        )

    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        text = transcribe_offline(
            audio_path=tmp_path,
            hotwords=hotwords,
            enable_punctuation=enable_punctuation,
            enable_itn=enable_itn,
            enable_vad=enable_vad,
        )
    except Exception as exc:
        logger.exception("识别失败")
        raise HTTPException(status_code=500, detail=f"transcribe error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return {"text": text, "sample_rate": sample_rate}


@app.post("/asr/punc")
async def asr_punc(
    request: Request,
    payload: dict,
) -> dict:
    """仅打标点。{text, mode: offline|realtime} -> {text}"""
    _check_internal_token(request)

    text = (payload.get("text") or "").strip()
    if not text:
        return {"text": ""}

    realtime = (payload.get("mode") or "offline").lower() == "realtime"
    try:
        punc_inst = _get_punc_model(realtime=realtime)
        result = punc_inst.generate(input=text, cache={})
        if result and len(result):
            return {"text": result[0].get("text", text)}
    except Exception as exc:
        logger.exception("PUNC 失败")
        raise HTTPException(status_code=500, detail=f"punc error: {exc}")
    return {"text": text}


# ---------------------------------------------------------------------------
# WebSocket 流式端点
#
# 协议(网关 ↔ 子服务):
#   client -> server:
#     - 首帧 JSON: {op:"start", chunk_size:[0,10,5], encoder_lookback:4,
#                   decoder_lookback:1, enable_realtime_punc:false,
#                   sample_rate:16000, format:"pcm"}
#     - 二进制 PCM int16 帧 = 普通 chunk (is_final=False)
#     - JSON {op:"flush"}    = is_final=True 但不带音频(强制 flush cache)
#     - JSON {op:"close"}    = 优雅结束
#
#   server -> client:
#     - JSON {type:"started"}                  # 收到 start 后
#     - JSON {type:"partial", text, text_punc, is_silence}  # 每个 chunk 后
#     - JSON {type:"flushed", text}            # flush 完成后
#     - JSON {type:"error", message}           # 出错
#
# 约定:
#   - 网关侧负责 PCM 完整 chunk 切分、句子边界状态机、远场过滤、
#     离线 PUNC、ITN — 子服务只做 generate + (可选)实时 PUNC。
#   - 鉴权: 通过 query string ?token=... 或 sec-websocket-protocol 头携带
#     X-Internal-Token 不便,这里走 query 参数 token。
# ---------------------------------------------------------------------------


import io as _io  # noqa: E402  (放这里避开顶层 import)
import numpy as _np  # noqa: E402


def _ws_check_token(websocket: WebSocket) -> bool:
    if not INTERNAL_SERVICE_TOKEN:
        return True
    token = websocket.query_params.get("token") or ""
    if token != INTERNAL_SERVICE_TOKEN:
        return False
    return True


@app.websocket("/asr/stream/v1")
async def asr_stream(websocket: WebSocket) -> None:
    if not _ws_check_token(websocket):
        await websocket.close(code=4401, reason="invalid internal token")
        return

    await websocket.accept()

    if ASR_MODEL_MODE not in ("all", "realtime"):
        await websocket.send_json(
            {
                "type": "error",
                "message": f"realtime model not loaded (mode={ASR_MODEL_MODE})",
            }
        )
        await websocket.close()
        return

    started = False
    chunk_size = [0, 10, 5]
    encoder_lookback = 4
    decoder_lookback = 1
    sample_rate = 16000
    audio_format = "pcm"
    enable_realtime_punc = False
    audio_cache: dict = {}
    punc_cache: dict = {}

    try:
        while True:
            msg = await websocket.receive()

            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "invalid json"}
                    )
                    continue

                op = payload.get("op")
                if op == "start":
                    chunk_size = payload.get("chunk_size", chunk_size)
                    encoder_lookback = int(
                        payload.get("encoder_lookback", encoder_lookback)
                    )
                    decoder_lookback = int(
                        payload.get("decoder_lookback", decoder_lookback)
                    )
                    sample_rate = int(payload.get("sample_rate", sample_rate))
                    audio_format = (payload.get("format") or "pcm").lower()
                    enable_realtime_punc = bool(
                        payload.get("enable_realtime_punc", False)
                    )
                    # 预加载实时 PUNC 模型避免首次 chunk 长等待
                    if enable_realtime_punc:
                        try:
                            _get_punc_model(realtime=True)
                        except Exception as exc:
                            logger.warning("实时 PUNC 模型加载失败: %s", exc)
                    started = True
                    await websocket.send_json({"type": "started"})

                elif op == "flush":
                    if not started:
                        await websocket.send_json(
                            {"type": "error", "message": "not started"}
                        )
                        continue

                    empty = _np.array([], dtype=_np.float32)
                    text = transcribe_realtime_chunk(
                        empty,
                        audio_cache,
                        is_final=True,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=encoder_lookback,
                        decoder_chunk_look_back=decoder_lookback,
                    )
                    await websocket.send_json({"type": "flushed", "text": text})
                    # flush 后清空 cache, 下一个 sentence 重新开始
                    audio_cache.clear()
                    punc_cache.clear()

                elif op == "close":
                    break

                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"unknown op: {op}"}
                    )

            elif "bytes" in msg:
                if not started:
                    await websocket.send_json(
                        {"type": "error", "message": "not started"}
                    )
                    continue

                audio_bytes = msg["bytes"]
                try:
                    if audio_format == "pcm":
                        audio_array = (
                            _np.frombuffer(audio_bytes, dtype=_np.int16).astype(
                                _np.float32
                            )
                            / 32768.0
                        )
                    elif audio_format == "wav":
                        import soundfile as sf

                        audio_array, _ = sf.read(_io.BytesIO(audio_bytes))
                        audio_array = _np.asarray(audio_array, dtype=_np.float32)
                    else:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": f"unsupported format: {audio_format}",
                            }
                        )
                        continue
                except Exception as exc:
                    await websocket.send_json(
                        {"type": "error", "message": f"audio decode: {exc}"}
                    )
                    continue

                # 子服务只做静音判定的轻量信号(便于网关侧统计),
                # 真正的远场过滤在网关侧已经做了
                is_silence = bool(
                    len(audio_array) > 0
                    and float(_np.max(_np.abs(audio_array))) < 0.002
                )

                try:
                    text_raw = transcribe_realtime_chunk(
                        audio_array,
                        audio_cache,
                        is_final=False,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=encoder_lookback,
                        decoder_chunk_look_back=decoder_lookback,
                    )
                except Exception as exc:
                    logger.exception("realtime chunk failed")
                    await websocket.send_json(
                        {"type": "error", "message": f"asr error: {exc}"}
                    )
                    continue

                text_punc = ""
                if enable_realtime_punc and text_raw:
                    try:
                        text_punc = punc_realtime_apply(text_raw, punc_cache)
                    except Exception as exc:
                        logger.warning("实时 PUNC 失败: %s", exc)

                await websocket.send_json(
                    {
                        "type": "partial",
                        "text": text_raw,
                        "text_punc": text_punc,
                        "is_silence": is_silence,
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
        port=int(os.getenv("PORT", "8001")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
