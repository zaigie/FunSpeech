# -*- coding: utf-8 -*-
"""
OpenAI-compatible audio API routes.

Gateway-owned compatibility surface:
- POST /v1/audio/speech
- POST /v1/audio/transcriptions
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from starlette.background import BackgroundTask

from ...core.config import settings
from ...core.executor import run_sync
from ...core.exceptions import (
    APIException,
    DefaultServerErrorException,
)
from ...core.security import validate_bearer_token, mask_sensitive_data
from ...models.tts import OpenAITTSRequest
from ...utils.common import generate_task_id, clean_text_for_tts
from ...utils.audio import (
    cleanup_temp_file,
    generate_temp_audio_path,
    get_audio_duration,
    normalize_audio_for_asr,
    resample_audio_array,
    save_audio_to_temp_file,
)
from ...services.asr.manager import get_model_manager
from ...services.tts.engine import get_tts_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio", tags=["OpenAI Audio"])


OPENAI_TTS_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
OPENAI_TTS_STREAM_FORMATS = {"audio", "sse"}
OPENAI_ASR_RESPONSE_FORMATS = {
    "json",
    "text",
    "srt",
    "verbose_json",
    "vtt",
    "diarized_json",
}
OPENAI_ASR_MODEL_ALIASES = {
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-transcribe-2025-12-15",
    "gpt-4o-transcribe-diarize",
}
OPENAI_ASR_INPUT_SUFFIXES = {
    ".flac",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".ogg",
    ".wav",
    ".webm",
}

TTS_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "application/octet-stream",
}


class OpenAICompatibleError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code
        super().__init__(message)


def _openai_error_response(exc: OpenAICompatibleError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


def _api_exception_to_openai(exc: APIException) -> OpenAICompatibleError:
    if exc.status_code == 40000001:
        return OpenAICompatibleError(
            exc.message,
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    if exc.status_code >= 50000000:
        return OpenAICompatibleError(
            exc.message,
            status_code=500,
            error_type="server_error",
            code=exc.error_code,
        )
    return OpenAICompatibleError(
        exc.message,
        status_code=400,
        error_type="invalid_request_error",
        code=exc.error_code,
    )


def _validate_openai_auth(request: Request, task_id: str) -> str:
    result, content = validate_bearer_token(request, task_id)
    if not result:
        raise OpenAICompatibleError(
            content,
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    return content


def _cleanup_files(paths: List[Optional[str]]) -> None:
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        cleanup_temp_file(path)


def _extract_voice_id(voice: Any) -> str:
    if isinstance(voice, str):
        value = voice.strip()
    elif isinstance(voice, dict):
        value = str(voice.get("id") or "").strip()
    else:
        value = ""
    if not value:
        raise OpenAICompatibleError(
            "voice must be a non-empty string or an object with an id",
            param="voice",
        )
    return value


def _normalize_tts_format(response_format: Optional[str]) -> str:
    fmt = (response_format or "mp3").lower()
    if fmt not in OPENAI_TTS_FORMATS:
        supported = ", ".join(sorted(OPENAI_TTS_FORMATS))
        raise OpenAICompatibleError(
            f"Unsupported response_format: {fmt}. Supported formats: {supported}",
            param="response_format",
        )
    return fmt


def _normalize_stream_format(stream_format: Optional[str]) -> Optional[str]:
    if stream_format is None:
        return None
    fmt = stream_format.lower()
    if fmt not in OPENAI_TTS_STREAM_FORMATS:
        supported = ", ".join(sorted(OPENAI_TTS_STREAM_FORMATS))
        raise OpenAICompatibleError(
            f"Unsupported stream_format: {fmt}. Supported formats: {supported}",
            param="stream_format",
        )
    return fmt


def _audio_array_to_pcm_i16_bytes(audio_array: np.ndarray) -> bytes:
    if audio_array is None or audio_array.size == 0:
        return b""
    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def _convert_wav_to_openai_format(wav_path: str, fmt: str, task_id: str) -> str:
    if fmt == "wav":
        return wav_path

    suffix = ".opus" if fmt == "opus" else f".{fmt}"
    output_path = generate_temp_audio_path(f"openai_{task_id}", suffix)

    if fmt == "pcm":
        audio, _ = sf.read(wav_path, dtype="float32", always_2d=False)
        with open(output_path, "wb") as fp:
            fp.write(_audio_array_to_pcm_i16_bytes(audio))
        return output_path

    ffmpeg_args = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame"],
        "aac": ["-f", "adts", "-codec:a", "aac"],
        "opus": ["-f", "opus", "-codec:a", "libopus"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
    }[fmt]

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                wav_path,
                *ffmpeg_args,
                output_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DefaultServerErrorException(
            f"Failed to convert audio to {fmt}: {exc}"
        ) from exc
    return output_path


async def _create_speech_file_response(
    request_body: OpenAITTSRequest,
    task_id: str,
    fmt: str,
    voice: str,
) -> FileResponse:
    output_path = None
    response_path = None
    try:
        clean_text = clean_text_for_tts(request_body.input)
        tts_engine = get_tts_engine()

        output_path = await run_sync(
            tts_engine.synthesize_speech,
            clean_text,
            voice,
            request_body.speed,
            "wav",
            22050,
            50,
            request_body.instructions or "",
        )
        response_path = await run_sync(_convert_wav_to_openai_format, output_path, fmt, task_id)

        cleanup_targets = [response_path]
        if response_path != output_path:
            cleanup_targets.append(output_path)

        return FileResponse(
            response_path,
            media_type=TTS_MEDIA_TYPES[fmt],
            filename=f"speech_{task_id}.{fmt}",
            headers={
                "X-Request-ID": task_id,
                "OpenAI-Processing-Ms": "0",
            },
            background=BackgroundTask(_cleanup_files, cleanup_targets),
        )
    except Exception:
        _cleanup_files([response_path, output_path])
        raise


async def _iter_tts_pcm_chunks(
    tts_engine: Any,
    *,
    text: str,
    voice: str,
    speed: float,
    prompt: str,
    sample_rate: int = 24000,
) -> AsyncGenerator[bytes, None]:
    async for audio_array, native_sr in tts_engine.iter_stream_audio_chunks(
        text=text,
        voice=voice,
        speed=speed,
        prompt=prompt,
    ):
        audio_array = resample_audio_array(audio_array, int(native_sr), sample_rate)
        chunk = _audio_array_to_pcm_i16_bytes(audio_array)
        if chunk:
            yield chunk


async def _iter_tts_sse_chunks(
    tts_engine: Any,
    *,
    text: str,
    voice: str,
    speed: float,
    prompt: str,
) -> AsyncGenerator[bytes, None]:
    async for chunk in _iter_tts_pcm_chunks(
        tts_engine,
        text=text,
        voice=voice,
        speed=speed,
        prompt=prompt,
    ):
        encoded = base64.b64encode(chunk).decode("ascii")
        yield _sse_payload(
            {
                "type": "speech.audio.delta",
                "audio": encoded,
                "delta": encoded,
            }
        )
    yield _sse_payload({"type": "speech.audio.done"})


@router.post("/speech", summary="Create speech")
async def create_speech(request_body: OpenAITTSRequest, request: Request):
    """OpenAI-compatible text-to-speech endpoint."""
    task_id = generate_task_id("openai")
    try:
        token = _validate_openai_auth(request, task_id)
        logger.debug(
            "[%s] OpenAI auth ok, token=%s",
            task_id,
            mask_sensitive_data(token) if token != "optional" else "optional",
        )

        fmt = _normalize_tts_format(request_body.response_format)
        stream_format = _normalize_stream_format(request_body.stream_format)
        voice = _extract_voice_id(request_body.voice)

        if stream_format:
            clean_text = clean_text_for_tts(request_body.input)
            tts_engine = get_tts_engine()
            iter_stream_audio_chunks = getattr(tts_engine, "iter_stream_audio_chunks", None)
            if callable(iter_stream_audio_chunks) and fmt == "pcm":
                if stream_format == "sse":
                    return StreamingResponse(
                        _iter_tts_sse_chunks(
                            tts_engine,
                            text=clean_text,
                            voice=voice,
                            speed=request_body.speed,
                            prompt=request_body.instructions or "",
                        ),
                        media_type="text/event-stream",
                        headers={"X-Request-ID": task_id},
                    )
                return StreamingResponse(
                    _iter_tts_pcm_chunks(
                        tts_engine,
                        text=clean_text,
                        voice=voice,
                        speed=request_body.speed,
                        prompt=request_body.instructions or "",
                    ),
                    media_type=TTS_MEDIA_TYPES["pcm"],
                    headers={"X-Request-ID": task_id},
                )

            logger.info(
                "[%s] Falling back to file response for speech stream_format=%s response_format=%s",
                task_id,
                stream_format,
                fmt,
            )

        return await _create_speech_file_response(request_body, task_id, fmt, voice)

    except OpenAICompatibleError as exc:
        logger.warning("[%s] OpenAI speech request error: %s", task_id, exc.message)
        return _openai_error_response(exc)
    except APIException as exc:
        logger.error("[%s] OpenAI speech API exception: %s", task_id, exc.message)
        return _openai_error_response(_api_exception_to_openai(exc))
    except Exception as exc:
        logger.error("[%s] OpenAI speech server error: %s", task_id, exc, exc_info=True)
        return _openai_error_response(
            OpenAICompatibleError(
                f"Internal server error: {exc}",
                status_code=500,
                error_type="server_error",
                code="internal_error",
            )
        )


def _form_first(form: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = form.get(name)
        if value is not None:
            return value
    return default


def _form_list(form: Any, *names: str) -> List[str]:
    values: List[str] = []
    for name in names:
        try:
            values.extend(str(v) for v in form.getlist(name))
        except AttributeError:
            value = form.get(name)
            if value is not None:
                values.append(str(value))
    return [v for v in values if v]


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_asr_response_format(value: Optional[str]) -> str:
    fmt = (value or "json").lower()
    if fmt not in OPENAI_ASR_RESPONSE_FORMATS:
        supported = ", ".join(sorted(OPENAI_ASR_RESPONSE_FORMATS))
        raise OpenAICompatibleError(
            f"Unsupported response_format: {fmt}. Supported formats: {supported}",
            param="response_format",
        )
    return fmt


def _upload_suffix(filename: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in OPENAI_ASR_INPUT_SUFFIXES:
        return suffix
    return ".wav"


def _resolve_asr_model_id(model: str, manager: Any) -> Optional[str]:
    available = {item["id"] for item in manager.list_models()}
    if model in available:
        return model
    if model in OPENAI_ASR_MODEL_ALIASES:
        return None
    # Be lenient for OpenAI-compatible clients that pass deployment aliases.
    logger.info("OpenAI ASR model %s is not local; using default ASR model", model)
    return None


async def _save_openai_upload(upload: UploadFile, task_id: str) -> str:
    audio_data = await upload.read()
    if not audio_data:
        raise OpenAICompatibleError("Uploaded file is empty", param="file")
    if len(audio_data) > settings.MAX_AUDIO_SIZE:
        max_size_mb = settings.MAX_AUDIO_SIZE // 1024 // 1024
        raise OpenAICompatibleError(
            f"Audio file is too large. Maximum size is {max_size_mb}MB",
            param="file",
        )
    return save_audio_to_temp_file(audio_data, _upload_suffix(upload.filename))


async def _run_transcription(
    *,
    upload: UploadFile,
    model: str,
    prompt: str,
    language: Optional[str],
    task_id: str,
) -> Tuple[str, float]:
    audio_path = None
    normalized_audio_path = None
    try:
        audio_path = await _save_openai_upload(upload, task_id)
        normalized_audio_path = await run_sync(normalize_audio_for_asr, audio_path, 16000)
        try:
            duration = await run_sync(get_audio_duration, normalized_audio_path)
        except Exception:
            duration = 0.0

        manager = get_model_manager()
        model_id = _resolve_asr_model_id(model, manager)
        asr_engine = manager.get_asr_engine(model_id)

        result_text = await run_sync(
            asr_engine.transcribe_file,
            audio_path=normalized_audio_path,
            hotwords=prompt or "",
            enable_punctuation=True,
            enable_itn=True,
            enable_vad=False,
            sample_rate=16000,
            dolphin_lang_sym=language or "zh",
            dolphin_region_sym="SHANGHAI",
        )
        return result_text, float(duration or 0.0)
    finally:
        _cleanup_files(
            [
                audio_path,
                normalized_audio_path if normalized_audio_path != audio_path else None,
            ]
        )


def _format_seconds_srt(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_seconds_vtt(seconds: float) -> str:
    return _format_seconds_srt(seconds).replace(",", ".")


def _caption_response(text: str, duration: float, fmt: str) -> Response:
    end = max(duration, 0.001)
    if fmt == "srt":
        body = f"1\n00:00:00,000 --> {_format_seconds_srt(end)}\n{text}\n"
    else:
        body = f"WEBVTT\n\n00:00:00.000 --> {_format_seconds_vtt(end)}\n{text}\n"
    return Response(content=body, media_type="text/plain; charset=utf-8")


def _verbose_json(text: str, duration: float, language: Optional[str]) -> Dict[str, Any]:
    segment = {
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": float(duration or 0.0),
        "text": text,
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
    }
    return {
        "task": "transcribe",
        "language": language or "unknown",
        "duration": float(duration or 0.0),
        "text": text,
        "segments": [segment] if text else [],
    }


def _diarized_json(text: str, duration: float) -> Dict[str, Any]:
    return {
        "task": "transcribe",
        "duration": float(duration or 0.0),
        "text": text,
        "segments": [
            {
                "id": "segment_0",
                "type": "transcript.text.segment",
                "speaker": "A",
                "start": 0.0,
                "end": float(duration or 0.0),
                "text": text,
            }
        ]
        if text
        else [],
    }


def _transcription_response(
    *,
    text: str,
    duration: float,
    language: Optional[str],
    response_format: str,
) -> Response:
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format in {"srt", "vtt"}:
        return _caption_response(text, duration, response_format)
    if response_format == "verbose_json":
        return JSONResponse(_verbose_json(text, duration, language))
    if response_format == "diarized_json":
        return JSONResponse(_diarized_json(text, duration))
    return JSONResponse({"text": text})


def _sse_payload(payload: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


async def _transcription_sse(text: str) -> AsyncGenerator[bytes, None]:
    if text:
        yield _sse_payload({"type": "transcript.text.delta", "delta": text})
    yield _sse_payload({"type": "transcript.text.done", "text": text})


@router.post("/transcriptions", summary="Create transcription")
async def create_transcription(request: Request):
    """OpenAI-compatible audio transcription endpoint."""
    task_id = generate_task_id("openai_asr")
    try:
        _validate_openai_auth(request, task_id)
        form = await request.form()

        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise OpenAICompatibleError("Missing required multipart file field: file", param="file")

        model = str(_form_first(form, "model", default="whisper-1"))
        prompt = str(_form_first(form, "prompt", default=""))
        language_raw = _form_first(form, "language", default=None)
        language = str(language_raw) if language_raw else None
        response_format = _normalize_asr_response_format(
            str(_form_first(form, "response_format", default="json"))
        )
        stream = _parse_bool(_form_first(form, "stream", default=False))

        # Accepted for SDK compatibility. Current backends do not expose these details.
        _form_list(form, "include", "include[]")
        _form_list(form, "timestamp_granularities", "timestamp_granularities[]")
        _form_first(form, "temperature", default=None)
        _form_first(form, "chunking_strategy", default=None)
        _form_list(form, "known_speaker_names", "known_speaker_names[]")
        _form_list(form, "known_speaker_references", "known_speaker_references[]")

        text, duration = await _run_transcription(
            upload=upload,
            model=model,
            prompt=prompt,
            language=language,
            task_id=task_id,
        )

        if stream:
            return StreamingResponse(
                _transcription_sse(text),
                media_type="text/event-stream",
                headers={"X-Request-ID": task_id},
            )

        return _transcription_response(
            text=text,
            duration=duration,
            language=language,
            response_format=response_format,
        )

    except OpenAICompatibleError as exc:
        logger.warning("[%s] OpenAI transcription request error: %s", task_id, exc.message)
        return _openai_error_response(exc)
    except APIException as exc:
        logger.error("[%s] OpenAI transcription API exception: %s", task_id, exc.message)
        return _openai_error_response(_api_exception_to_openai(exc))
    except Exception as exc:
        logger.error("[%s] OpenAI transcription server error: %s", task_id, exc, exc_info=True)
        return _openai_error_response(
            OpenAICompatibleError(
                f"Internal server error: {exc}",
                status_code=500,
                error_type="server_error",
                code="internal_error",
            )
        )
