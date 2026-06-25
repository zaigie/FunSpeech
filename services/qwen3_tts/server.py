# -*- coding: utf-8 -*-
"""Qwen3-TTS 开源本地推理子服务。

默认按 Base Voice Clone 模型工作,接口对齐 services/cosyvoice:
    GET    /health
    POST   /tts/file
    WS     /tts/stream
    GET    /voices
    GET    /voices/{name}
    POST   /voices
    DELETE /voices/{name}
    POST   /voices/refresh
    POST   /voices/reload

CosyVoice 的 SFT 模型不支持 clone 时会拒绝音色写入;这里也采用同样
策略: 当前网关集成只接受 Qwen3-TTS Base 模型,并只暴露可持久化的
Base Clone 音色库。
"""

from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from fastapi.responses import JSONResponse, Response


logger = logging.getLogger("qwen3_tts_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


PORT = int(os.getenv("PORT", "8005"))
INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")
MODEL_ID = os.getenv("QWEN3_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
DEVICE = os.getenv("QWEN3_TTS_DEVICE", "cuda:0")
DTYPE = os.getenv("QWEN3_TTS_DTYPE", "bfloat16").lower()
ATTN_IMPLEMENTATION = os.getenv("QWEN3_TTS_ATTN_IMPLEMENTATION", "sdpa")
DEFAULT_LANGUAGE = os.getenv("QWEN3_TTS_LANGUAGE", "Auto")
GPU_INFERENCE_CONCURRENCY = int(os.getenv("QWEN3_TTS_GPU_CONCURRENCY", "1"))
STREAM_CHUNK_SEC = float(os.getenv("QWEN3_TTS_STREAM_CHUNK_SEC", "0.2"))
X_VECTOR_ONLY_MODE = os.getenv("QWEN3_TTS_X_VECTOR_ONLY_MODE", "false").lower() == "true"

VOICES_DIR = Path(
    os.getenv("QWEN3_TTS_VOICES_DIR", os.getenv("VOICES_DIR", "/app/qwen3_voices"))
)
PROMPT_DIR = VOICES_DIR / "prompts"
REGISTRY_FILE = VOICES_DIR / "voice_registry.json"

_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9一-鿿._-]{1,64}$")

_model = None
_model_task = "voice_clone"
_supported_speakers: List[str] = []
_supported_languages: List[str] = []
_registry: Dict[str, Any] = {"version": "qwen3-tts-v1", "voices": {}}
_prompt_cache: Dict[str, List[Any]] = {}
_load_failed = False
_load_error_msg = ""
_load_lock = threading.Lock()
_voice_lock = threading.Lock()
_gpu_sem: Optional[asyncio.Semaphore] = None


def _detect_task(model_id: str) -> str:
    lowered = model_id.lower()
    if "base" in lowered:
        return "voice_clone"
    return "unsupported"


def _torch_dtype():
    import torch

    if DTYPE in ("bf16", "bfloat16"):
        return torch.bfloat16
    if DTYPE in ("fp16", "float16", "half"):
        return torch.float16
    if DTYPE in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16


def _validate_voice_name(name: str) -> None:
    if not name or not _VOICE_NAME_RE.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "invalid voice name: only A-Za-z0-9, 中文, '.', '_', '-' allowed, "
                "length 1-64"
            ),
        )


def _ensure_voices_dirs() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    _ensure_voices_dirs()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fp:
        fp.write(data)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def _load_registry_from_disk() -> Dict[str, Any]:
    if not REGISTRY_FILE.exists():
        return {"version": "qwen3-tts-v1", "voices": {}}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError("registry root is not object")
        data.setdefault("version", "qwen3-tts-v1")
        data.setdefault("voices", {})
        return data
    except Exception as exc:
        logger.warning("加载 Qwen3-TTS voice registry 失败: %s", exc)
        return {"version": "qwen3-tts-v1", "voices": {}}


def _save_registry() -> None:
    now = datetime.datetime.now().isoformat()
    _registry["updated_at"] = now
    if not _registry.get("created_at"):
        _registry["created_at"] = now
    payload = json.dumps(_registry, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(REGISTRY_FILE, payload)


def _prompt_file(name: str) -> Path:
    return PROMPT_DIR / f"{name}.pt"


def _tensor_to_cpu(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu()
    return value


def _prompt_items_to_payload(items: List[Any]) -> Dict[str, Any]:
    payload_items = []
    for item in items:
        payload_items.append(
            {
                "ref_code": _tensor_to_cpu(item.ref_code),
                "ref_spk_embedding": _tensor_to_cpu(item.ref_spk_embedding),
                "x_vector_only_mode": bool(item.x_vector_only_mode),
                "icl_mode": bool(item.icl_mode),
                "ref_text": item.ref_text,
            }
        )
    return {"version": "qwen3-tts-prompt-v1", "items": payload_items}


def _move_tensor_to_runtime_device(value: Any) -> Any:
    if value is None or not hasattr(value, "to"):
        return value
    if DEVICE == "cpu" or DEVICE.startswith("cuda"):
        return value.to(DEVICE)
    return value


def _payload_to_prompt_items(payload: Dict[str, Any]) -> List[Any]:
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    items = []
    for item in payload.get("items", []):
        items.append(
            VoiceClonePromptItem(
                ref_code=_move_tensor_to_runtime_device(item.get("ref_code")),
                ref_spk_embedding=_move_tensor_to_runtime_device(
                    item["ref_spk_embedding"]
                ),
                x_vector_only_mode=bool(item.get("x_vector_only_mode", False)),
                icl_mode=bool(item.get("icl_mode", True)),
                ref_text=item.get("ref_text"),
            )
        )
    return items


def _save_prompt_items(name: str, items: List[Any]) -> None:
    import torch

    buf = io.BytesIO()
    torch.save(_prompt_items_to_payload(items), buf)
    _atomic_write_bytes(_prompt_file(name), buf.getvalue())
    _prompt_cache[name] = items


def _load_prompt_items(name: str) -> List[Any]:
    if name in _prompt_cache:
        return _prompt_cache[name]
    path = _prompt_file(name)
    if not path.exists():
        raise RuntimeError(f"voice prompt not found: {name}")
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    items = _payload_to_prompt_items(payload)
    _prompt_cache[name] = items
    return items


def _load_model() -> None:
    global _model, _model_task, _supported_speakers, _supported_languages, _registry

    with _load_lock:
        if _model is not None:
            return

        detected_task = _detect_task(MODEL_ID)
        if detected_task != "voice_clone":
            raise RuntimeError(
                "Qwen3-TTS service currently supports only Base Clone models; "
                f"got model={MODEL_ID!r} task={detected_task}"
            )

        from qwen_tts import Qwen3TTSModel

        kwargs: Dict[str, Any] = {
            "device_map": DEVICE,
            "dtype": _torch_dtype(),
        }
        if ATTN_IMPLEMENTATION:
            kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

        logger.info(
            "加载 Qwen3-TTS: model=%s, device=%s, dtype=%s, attn=%s",
            MODEL_ID,
            DEVICE,
            DTYPE,
            ATTN_IMPLEMENTATION,
        )
        _model = Qwen3TTSModel.from_pretrained(MODEL_ID, **kwargs)
        _model_task = detected_task

        try:
            speakers = _model.get_supported_speakers() or []
            _supported_speakers = list(speakers)
        except Exception:
            _supported_speakers = []
        try:
            languages = _model.get_supported_languages() or []
            _supported_languages = list(languages)
        except Exception:
            _supported_languages = []

        _registry = _load_registry_from_disk()

        logger.info(
            "Qwen3-TTS 加载完成: task=%s speakers=%s languages=%s voices=%d",
            _model_task,
            _supported_speakers,
            _supported_languages,
            len(_registry.get("voices", {})),
        )


def _check_token(request: Request) -> None:
    if not INTERNAL_SERVICE_TOKEN:
        return
    if request.headers.get("X-Internal-Token", "") != INTERNAL_SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="invalid internal token")


def _ws_check_token(websocket: WebSocket) -> bool:
    if not INTERNAL_SERVICE_TOKEN:
        return True
    return websocket.query_params.get("token", "") == INTERNAL_SERVICE_TOKEN


def _match_supported(value: str, supported: List[str]) -> str:
    lowered = value.lower()
    for item in supported:
        if item.lower() == lowered:
            return item
    return value


def _resolve_language(language: Optional[str] = None) -> Optional[str]:
    value = (language or DEFAULT_LANGUAGE or "Auto").strip()
    if not value or value.lower() in ("auto", "none", "null"):
        return None
    return _match_supported(value, _supported_languages)


def _clone_voices() -> List[str]:
    return [
        name
        for name in (_registry.get("voices") or {})
        if _prompt_file(name).exists()
    ]


def _voices() -> List[str]:
    return _clone_voices()


def _voices_info(sample_rate: int = 24000) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    registry = _registry.get("voices") or {}
    for voice in _clone_voices():
        record = registry.get(voice) or {}
        out[voice] = {
            "name": voice,
            "type": "clone",
            "language": record.get("language", "multilingual"),
            "gender": record.get("gender", "unknown"),
            "description": f"Qwen3-TTS Base clone voice: {voice}",
            "sample_rate": sample_rate,
            "available": True,
            "reference_text": record.get("reference_text", ""),
            "audio_file": record.get("audio_file", ""),
            "audio_duration": record.get("audio_duration"),
            "added_at": record.get("added_at", ""),
            "x_vector_only_mode": record.get("x_vector_only_mode", False),
        }
    return out


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _audio_item_to_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _estimate_sentences(text: str, sample_count: int, sample_rate: int) -> List[Dict[str, str]]:
    parts = [p for p in re.split(r"(?<=[。！？.!?])\s*", text) if p]
    if not parts:
        return []
    total_chars = max(sum(len(p) for p in parts), 1)
    total_ms = sample_count / sample_rate * 1000
    cursor = 0.0
    out: List[Dict[str, str]] = []
    for part in parts:
        dur = total_ms * (len(part) / total_chars)
        out.append(
            {
                "text": part,
                "begin_time": str(int(cursor)),
                "end_time": str(int(cursor + dur)),
            }
        )
        cursor += dur
    return out


def _ensure_base_clone_loaded() -> None:
    if _model_task != "voice_clone":
        raise RuntimeError(
            "Qwen3-TTS voice CRUD requires a Base model; "
            f"current model task is {_model_task}"
        )
    if _model is None:
        raise RuntimeError("model not loaded")


def _voice_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def voice_add(name: str, prompt_text: str, wav_path: Path) -> Dict[str, Any]:
    _ensure_base_clone_loaded()
    if not X_VECTOR_ONLY_MODE and not prompt_text.strip():
        raise ValueError("prompt_text required for Qwen3-TTS Base ICL clone")

    duration = _voice_audio_duration(wav_path)
    if duration < 1.0:
        raise ValueError(f"音频过短 ({duration:.2f}s), 至少 1 秒")
    if duration > 30.0:
        logger.warning("音频较长 (%.2fs), 建议 ≤30s", duration)

    with _voice_lock:
        prompt_items = _model.create_voice_clone_prompt(
            ref_audio=str(wav_path),
            ref_text=prompt_text.strip() or None,
            x_vector_only_mode=X_VECTOR_ONLY_MODE,
        )
        _save_prompt_items(name, prompt_items)

        record = {
            "name": name,
            "reference_text": prompt_text.strip(),
            "audio_file": wav_path.name,
            "prompt_file": _prompt_file(name).name,
            "file_size": os.path.getsize(wav_path),
            "audio_duration": duration,
            "x_vector_only_mode": X_VECTOR_ONLY_MODE,
            "added_at": datetime.datetime.now().isoformat(),
            "status": "active",
        }
        _registry["voices"][name] = record
        _save_registry()
        return record


def _unlink_file_under(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
    except (ValueError, OSError):
        logger.warning("跳过删除 VOICES_DIR 外路径: %s", path)
        return False
    if resolved.exists() and resolved.is_file():
        resolved.unlink()
        return True
    return False


def voice_remove(name: str) -> bool:
    with _voice_lock:
        removed = False
        record = _registry.get("voices", {}).pop(name, None)
        if record is not None:
            _save_registry()
            removed = True
        _prompt_cache.pop(name, None)

        paths = [_prompt_file(name), VOICES_DIR / f"{name}.txt"]
        audio_file = (record or {}).get("audio_file")
        if audio_file:
            paths.append(VOICES_DIR / Path(audio_file).name)

        for path in paths:
            if _unlink_file_under(path, VOICES_DIR):
                removed = True
        return removed


def voice_refresh_from_dir() -> Tuple[int, int]:
    _ensure_base_clone_loaded()
    _ensure_voices_dirs()

    pairs: List[Tuple[str, Path, Path]] = []
    for txt in VOICES_DIR.glob("*.txt"):
        wav = txt.with_suffix(".wav")
        if wav.exists():
            pairs.append((txt.stem, txt, wav))

    success = 0
    for name, txt_path, wav_path in pairs:
        if name in _registry.get("voices", {}) and _prompt_file(name).exists():
            continue
        try:
            _validate_voice_name(name)
            prompt_text = txt_path.read_text(encoding="utf-8").strip()
            voice_add(name, prompt_text, wav_path)
            success += 1
        except Exception as exc:
            logger.warning("注册 Qwen3-TTS 音色 %s 失败: %s", name, exc)
    return success, len(pairs)


def reload_voices_from_disk() -> Dict[str, Any]:
    global _registry, _prompt_cache
    with _voice_lock:
        _registry = _load_registry_from_disk()
        _prompt_cache = {}
        valid = [
            name
            for name in _registry.get("voices", {})
            if _prompt_file(name).exists()
        ]
    logger.info("Qwen3-TTS 音色已热重载: %d clone voices", len(valid))
    return {
        "clone_voices": len(valid),
        "registry_voices": len(_registry.get("voices", {})),
        "clone_loaded": _model_task == "voice_clone",
    }


def synthesize_offline(
    text: str,
    voice: str,
    speed: float,
    prompt: str,
    return_timestamps: bool,
    language: Optional[str] = None,
) -> Tuple[bytes, int, Optional[List[Dict[str, str]]]]:
    del speed  # qwen-tts generate_* 当前公开接口没有 speech speed 参数。
    if _model is None:
        raise RuntimeError("model not loaded")
    if _model_task != "voice_clone":
        raise RuntimeError(f"unsupported Qwen3-TTS task: {_model_task}")

    lang = _resolve_language(language)
    _validate_voice_name(voice)
    if voice not in _registry.get("voices", {}):
        raise RuntimeError(f"voice not found: {voice}")
    prompt_items = _load_prompt_items(voice)
    wavs, sr = _model.generate_voice_clone(
        text=text,
        language=lang,
        voice_clone_prompt=prompt_items,
        non_streaming_mode=True,
    )

    if not wavs:
        raise RuntimeError("inference produced no audio")
    audio = _audio_item_to_numpy(wavs[0])
    wav_bytes = _audio_to_wav_bytes(audio, int(sr))
    sentences = _estimate_sentences(text, len(audio), int(sr)) if return_timestamps else None
    return wav_bytes, int(sr), sentences


async def _run_inference(fn, **kwargs):
    sem = _get_gpu_semaphore()
    async with sem:
        return await asyncio.to_thread(fn, **kwargs)


def _get_gpu_semaphore() -> asyncio.Semaphore:
    global _gpu_sem
    if _gpu_sem is None:
        _gpu_sem = asyncio.Semaphore(max(1, GPU_INFERENCE_CONCURRENCY))
    return _gpu_sem


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _load_failed, _load_error_msg
    try:
        await asyncio.to_thread(_load_model)
        _get_gpu_semaphore()
    except Exception as exc:
        _load_failed = True
        _load_error_msg = str(exc)
        logger.error("Qwen3-TTS 模型预加载失败: %s", exc, exc_info=True)
    yield


app = FastAPI(title="funspeech-qwen3-tts-service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    clone_loaded = _model is not None and _model_task == "voice_clone"
    body = {
        "status": "healthy",
        "device": DEVICE,
        "model_id": MODEL_ID,
        "task": _model_task,
        "model_loaded": _model is not None,
        "sft_loaded": False,
        "clone_loaded": clone_loaded,
        "voices_dir": str(VOICES_DIR),
        "clone_voices": len(_clone_voices()),
        "speakers": _supported_speakers,
        "languages": _supported_languages,
    }
    if _load_failed:
        body["status"] = "unhealthy"
        body["error"] = _load_error_msg
        return JSONResponse(status_code=503, content=body)
    if _model is None:
        body["status"] = "starting"
        return JSONResponse(status_code=503, content=body)
    return body


@app.post("/tts/file")
async def tts_file(request: Request) -> Response:
    _check_token(request)
    body = await request.json()
    text = body.get("text") or ""
    voice = body.get("voice") or "中文女"
    speed = float(body.get("speed", 1.0))
    prompt = body.get("prompt") or ""
    return_timestamps = bool(body.get("return_timestamps", False))
    language = body.get("language")

    if not text:
        raise HTTPException(status_code=400, detail="text required")

    try:
        wav_bytes, native_sr, sentences = await _run_inference(
            synthesize_offline,
            text=text,
            voice=voice,
            speed=speed,
            prompt=prompt,
            return_timestamps=return_timestamps,
            language=language,
        )
    except Exception as exc:
        logger.exception("合成失败")
        raise HTTPException(status_code=500, detail=f"synthesis: {exc}")

    headers = {"X-Native-Sample-Rate": str(native_sr)}
    if sentences is not None:
        headers["X-Sentences"] = json.dumps(sentences, ensure_ascii=True)
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.get("/voices")
async def voices_list(request: Request) -> dict:
    _check_token(request)
    voices = _voices()
    return {
        "preset": [],
        "clone": voices,
        "all": voices,
        "registry": _registry.get("voices", {}),
        "info": _voices_info(),
        "speakers": _supported_speakers,
        "languages": _supported_languages,
    }


@app.get("/voices/{name}")
async def voice_info(request: Request, name: str) -> dict:
    _check_token(request)
    _validate_voice_name(name)
    record = _registry.get("voices", {}).get(name)
    if record is None or not _prompt_file(name).exists():
        raise HTTPException(status_code=404, detail=f"voice not found: {name}")
    return {"name": name, "type": "clone", **record}


@app.post("/voices")
async def voice_create(
    request: Request,
    name: str = Form(...),
    prompt_text: str = Form(...),
    audio: UploadFile = File(...),
) -> dict:
    _check_token(request)
    _validate_voice_name(name)
    _ensure_base_clone_loaded()
    _ensure_voices_dirs()

    suffix_raw = os.path.splitext(audio.filename or "voice.wav")[1] or ".wav"
    suffix = re.sub(r"[^A-Za-z0-9.]", "", suffix_raw)[:8] or ".wav"
    target = VOICES_DIR / f"{name}{suffix}"
    try:
        target_resolved = target.resolve()
        voices_resolved = VOICES_DIR.resolve()
        target_resolved.relative_to(voices_resolved)
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="invalid voice path")

    audio_bytes = await audio.read()
    with open(target, "wb") as fp:
        fp.write(audio_bytes)

    txt_path = VOICES_DIR / f"{name}.txt"
    txt_path.write_text(prompt_text.strip(), encoding="utf-8")

    try:
        record = await _run_inference(
            voice_add,
            name=name,
            prompt_text=prompt_text,
            wav_path=target,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("注册 Qwen3-TTS 音色失败")
        _prompt_cache.pop(name, None)
        for path in (_prompt_file(name), target, txt_path):
            _unlink_file_under(path, VOICES_DIR)
        raise HTTPException(status_code=400, detail=str(exc))
    return record


@app.delete("/voices/{name}")
async def voice_delete(request: Request, name: str) -> dict:
    _check_token(request)
    _validate_voice_name(name)
    ok = await asyncio.to_thread(voice_remove, name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"voice not found: {name}")
    return {"removed": name}


@app.post("/voices/refresh")
async def voices_refresh(request: Request) -> dict:
    _check_token(request)
    try:
        success, total = await _run_inference(voice_refresh_from_dir)
    except Exception as exc:
        logger.exception("刷新 Qwen3-TTS 音色失败")
        raise HTTPException(status_code=400, detail=str(exc))
    return {"added": success, "total": total}


@app.post("/voices/reload")
async def voices_reload(request: Request) -> dict:
    _check_token(request)
    return reload_voices_from_disk()


@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket) -> None:
    if not _ws_check_token(websocket):
        await websocket.close(code=4401, reason="invalid internal token")
        return

    await websocket.accept()
    try:
        first = await websocket.receive_text()
        params = json.loads(first)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"bad start: {exc}"})
        await websocket.close()
        return

    text = params.get("text") or ""
    voice = params.get("voice") or "中文女"
    speed = float(params.get("speed", 1.0))
    prompt = params.get("prompt") or ""
    if not text:
        await websocket.send_json({"type": "error", "message": "text required"})
        await websocket.close()
        return

    try:
        wav_bytes, native_sr, _ = await _run_inference(
            synthesize_offline,
            text=text,
            voice=voice,
            speed=speed,
            prompt=prompt,
            return_timestamps=False,
        )
        audio, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        await websocket.send_json({"type": "started", "sample_rate": native_sr})

        chunk_size = max(1, int(native_sr * STREAM_CHUNK_SEC))
        for start in range(0, len(audio), chunk_size):
            if websocket.client_state.name != "CONNECTED":
                break
            chunk = np.asarray(audio[start:start + chunk_size], dtype=np.float32)
            await websocket.send_bytes(chunk.tobytes())
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        logger.info("Qwen3-TTS WS 客户端断开")
    except Exception as exc:
        logger.exception("流式合成失败")
        try:
            await websocket.send_json({"type": "error", "message": f"synthesis: {exc}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
