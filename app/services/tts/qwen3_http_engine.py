# -*- coding: utf-8 -*-
"""Qwen3-TTS 子服务 HTTP 客户端引擎。

与 CosyVoiceHttpEngine 保持同一组网关侧方法, 但底层调用
services/qwen3_tts 子服务。Qwen3-TTS 模型依赖独立放在子服务 venv 中,
gateway 不直接 import torch / transformers / qwen_tts。
"""

from __future__ import annotations

import io
import json
import logging
import random
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
import soundfile as sf

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from ...utils.audio import generate_temp_audio_path, save_audio_array
from ..asr.http_engine import _get_httpx_client

logger = logging.getLogger(__name__)


def _split_urls(raw: str) -> List[str]:
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


class _HttpReplicaPool:
    def __init__(self, urls: List[str]):
        if not urls:
            raise ValueError("Qwen3TTSHttpEngine requires at least one replica URL")
        self._urls = urls
        self._active = [0] * len(urls)
        self._lock = threading.Lock()

    def acquire(self) -> Tuple[int, str]:
        with self._lock:
            min_count = min(self._active)
            candidates = [i for i, c in enumerate(self._active) if c == min_count]
            idx = random.choice(candidates)
            self._active[idx] += 1
            return idx, self._urls[idx]

    def release(self, idx: int) -> None:
        with self._lock:
            if 0 <= idx < len(self._active):
                self._active[idx] = max(0, self._active[idx] - 1)


class _Qwen3VoiceManager:
    """Qwen3-TTS Base clone 音色管理器 — HTTP 客户端版本。"""

    def __init__(self, engine: "Qwen3TTSHttpEngine"):
        self._engine = engine
        self._cached_lists: Optional[Dict[str, Any]] = None
        self._cache_lock = threading.Lock()

    def _fetch(self) -> Dict[str, Any]:
        with self._cache_lock:
            if self._cached_lists is None:
                self._cached_lists = self._engine._get_voices_listing()
            return self._cached_lists

    def _invalidate(self) -> None:
        with self._cache_lock:
            self._cached_lists = None

    def list_voices(self) -> List[str]:
        return self._fetch().get("all", [])

    def list_clone_voices(self) -> List[str]:
        return self._fetch().get("clone", [])

    def get_voice_info(self, name: str) -> Optional[Dict[str, Any]]:
        try:
            return self._engine._get_voice_info(name)
        except Exception:
            return None

    def is_voice_available(self, name: str) -> bool:
        return name in self.list_voices()

    def add_voice(self, voice_name: str, txt_file, wav_file) -> bool:
        from pathlib import Path

        txt_path = Path(txt_file)
        wav_path = Path(wav_file)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            self._engine._post_voice(voice_name, prompt_text, wav_path)
            self._invalidate()
            return True
        except Exception as exc:
            logger.warning("Qwen3-TTS add_voice 失败 %s: %s", voice_name, exc)
            return False

    def remove_voice(self, voice_name: str) -> bool:
        try:
            self._engine._delete_voice(voice_name)
            self._invalidate()
            return True
        except Exception as exc:
            logger.warning("Qwen3-TTS remove_voice 失败 %s: %s", voice_name, exc)
            return False

    def refresh_voices(self) -> Tuple[int, int]:
        try:
            r = self._engine._post_voices_refresh()
            self._invalidate()
            return r.get("added", 0), r.get("total", 0)
        except Exception as exc:
            logger.warning("Qwen3-TTS refresh_voices 失败: %s", exc)
            return 0, 0

    def add_all_voices(self) -> Tuple[int, int]:
        return self.refresh_voices()

    def get_registry_info(self) -> Dict[str, Any]:
        listing = self._fetch()
        return {
            "version": "qwen3-tts",
            "total_voices": len(listing.get("all", [])),
            "clone_voices": len(listing.get("clone", [])),
            "preset_voices": len(listing.get("preset", [])),
            "voices": listing.get("clone", []),
        }


class Qwen3TTSHttpEngine:
    """Qwen3-TTS 子服务客户端 facade。"""

    def __init__(self, urls: List[str], internal_token: str = "", timeout: float = 120.0):
        self._pool = _HttpReplicaPool(urls)
        self._urls = list(urls)
        self._primary_url = urls[0] if urls else ""
        self._timeout = timeout
        self._internal_token = internal_token
        self._headers: Dict[str, str] = {}
        if internal_token:
            self._headers["X-Internal-Token"] = internal_token
        self._voice_manager = _Qwen3VoiceManager(self)
        self._sft_loaded = False
        self._clone_loaded = False
        self._cached_health_at = 0.0

    def _broadcast_reload(self) -> None:
        targets = [u for u in self._urls if u != self._primary_url]
        if not targets:
            return
        for url in targets:
            try:
                r = _get_httpx_client().post(
                    f"{url}/voices/reload",
                    headers=self._headers,
                    timeout=min(self._timeout, 30.0),
                )
                if r.status_code != 200:
                    logger.warning(
                        "Qwen3-TTS 广播 reload 到 %s 返回 %s: %s",
                        url,
                        r.status_code,
                        r.text[:200],
                    )
            except httpx.HTTPError as exc:
                logger.warning("Qwen3-TTS 广播 reload 到 %s 失败: %s", url, exc)

    @property
    def voice_manager(self):
        return self._voice_manager

    @property
    def device(self) -> str:
        return f"remote:qwen3-tts:{','.join(self._pool._urls)}"

    def is_sft_model_loaded(self) -> bool:
        self._refresh_health_if_stale()
        return self._sft_loaded

    def is_clone_model_loaded(self) -> bool:
        self._refresh_health_if_stale()
        return self._clone_loaded

    def is_tts_model_loaded(self) -> bool:
        return self.is_sft_model_loaded() or self.is_clone_model_loaded()

    def is_model_loaded(self) -> bool:
        return self.is_tts_model_loaded()

    def synthesize_speech(
        self,
        text: str,
        voice: str = "中文女",
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int = 22050,
        volume: int = 50,
        prompt: str = "",
        return_timestamps: bool = False,
    ) -> Union[str, Tuple[str, Optional[List[Dict[str, Any]]]]]:
        wav_bytes, native_sr, sentences = self._post_tts_file(
            text=text,
            voice=voice,
            speed=speed,
            prompt=prompt,
            return_timestamps=return_timestamps,
        )

        audio, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        is_clone = voice in self._voice_manager.list_clone_voices()
        prefix = "clone_voice" if is_clone else "preset_voice"
        output_path = generate_temp_audio_path(prefix, f".{format}")
        save_audio_array(
            audio,
            output_path,
            sample_rate=sample_rate,
            format=format,
            original_sr=native_sr,
            volume=volume,
        )
        if return_timestamps:
            return output_path, sentences
        return output_path

    def synthesize_with_preset_voice(
        self,
        text: str,
        voice: str = "中文女",
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int = 22050,
        volume: int = 50,
        return_timestamps: bool = False,
    ) -> Union[str, Tuple[str, Optional[List[Dict[str, Any]]]]]:
        return self.synthesize_speech(
            text=text,
            voice=voice,
            speed=speed,
            format=format,
            sample_rate=sample_rate,
            volume=volume,
            prompt="",
            return_timestamps=return_timestamps,
        )

    def get_voices(self) -> List[str]:
        listing = self._get_voices_listing()
        mode = settings.TTS_MODEL_MODE.lower()
        if mode in ("sft", "preset", "custom", "customvoice", "voicedesign"):
            return listing.get("preset", [])
        if mode in ("clone", "base"):
            return listing.get("clone", [])
        return listing.get("all", [])

    def get_voices_info(self) -> Dict[str, Dict[str, Any]]:
        listing = self._get_voices_listing()
        info = listing.get("info", {}) or {}
        registry_voices = listing.get("registry", {}) or {}
        clone_voices = set(listing.get("clone", []))
        out: Dict[str, Dict[str, Any]] = {}
        for voice in self.get_voices():
            item = dict(info.get(voice, {}))
            item.setdefault("name", voice)
            if voice in clone_voices:
                record = registry_voices.get(voice) or {}
                item.setdefault("type", "clone")
                item.setdefault("reference_text", record.get("reference_text", ""))
                item.setdefault("audio_file", record.get("audio_file", ""))
                item.setdefault("added_at", record.get("added_at", ""))
            else:
                item.setdefault("type", "preset")
            item.setdefault("language", "multilingual")
            item.setdefault("gender", "unknown")
            item.setdefault("description", f"Qwen3-TTS voice: {voice}")
            item.setdefault("sample_rate", listing.get("sample_rate", 24000))
            item.setdefault("available", True)
            out[voice] = item
        return out

    def refresh_voices(self) -> None:
        self._voice_manager._invalidate()

    def get_replica_healths(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for url in self._urls:
            item: Dict[str, Any] = {"url": url, "ok": False}
            try:
                r = _get_httpx_client().get(
                    f"{url}/health",
                    timeout=5.0,
                    headers=self._headers,
                )
                item["status_code"] = r.status_code
                try:
                    body = r.json()
                except ValueError:
                    body = {"raw": r.text[:500]}
                item.update(body if isinstance(body, dict) else {"body": body})
                item["ok"] = r.status_code == 200
            except Exception as exc:
                item["error"] = str(exc)
            results.append(item)
        return results

    async def iter_stream_audio_chunks(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        prompt: str = "",
    ):
        idx, base_url = self._pool.acquire()
        ws_url = base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        ) + "/tts/stream"

        if self._headers.get("X-Internal-Token"):
            sep = "&" if "?" in ws_url else "?"
            ws_url = f"{ws_url}{sep}token={self._headers['X-Internal-Token']}"

        import asyncio as _asyncio

        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        ws = None
        try:
            ws = await connect(ws_url, open_timeout=self._timeout)
            await ws.send(
                json.dumps(
                    {
                        "text": text,
                        "voice": voice,
                        "speed": speed,
                        "prompt": prompt,
                    },
                    ensure_ascii=False,
                )
            )

            first = await _asyncio.wait_for(ws.recv(), timeout=self._timeout)
            if isinstance(first, (bytes, bytearray)):
                raise DefaultServerErrorException("qwen3-tts ws started 帧非文本")
            meta = json.loads(first)
            if meta.get("type") == "error":
                raise DefaultServerErrorException(
                    f"qwen3-tts stream error: {meta.get('message')}"
                )
            if meta.get("type") != "started":
                raise DefaultServerErrorException(
                    f"qwen3-tts stream unexpected first frame: {meta}"
                )
            native_sr = int(meta.get("sample_rate", 24000))

            while True:
                try:
                    msg = await _asyncio.wait_for(ws.recv(), timeout=self._timeout)
                except _asyncio.TimeoutError as exc:
                    raise DefaultServerErrorException("qwen3-tts stream recv 超时") from exc
                except ConnectionClosed:
                    return
                if isinstance(msg, (bytes, bytearray)):
                    chunk = np.frombuffer(msg, dtype=np.float32)
                    if chunk.size:
                        yield chunk.reshape(1, -1), native_sr
                else:
                    evt = json.loads(msg)
                    if evt.get("type") == "done":
                        return
                    if evt.get("type") == "error":
                        raise DefaultServerErrorException(
                            f"qwen3-tts stream error: {evt.get('message')}"
                        )
        finally:
            if ws is not None:
                try:
                    await ws.close()
                except ConnectionClosed:
                    pass
            self._pool.release(idx)

    def _refresh_health_if_stale(self) -> None:
        import time

        now = time.time()
        if now - self._cached_health_at < settings.SERVICE_HEALTHCHECK_INTERVAL:
            return
        self._cached_health_at = now

        for url in self._urls:
            try:
                r = _get_httpx_client().get(f"{url}/health", timeout=5.0, headers=self._headers)
                if r.status_code == 200:
                    body = r.json()
                    self._sft_loaded = bool(body.get("sft_loaded"))
                    self._clone_loaded = bool(body.get("clone_loaded"))
                    return
            except httpx.HTTPError:
                continue

    def _post_tts_file(
        self,
        text: str,
        voice: str,
        speed: float,
        prompt: str,
        return_timestamps: bool,
    ) -> Tuple[bytes, int, Optional[List[Dict[str, Any]]]]:
        idx, base_url = self._pool.acquire()
        try:
            resp = _get_httpx_client().post(
                f"{base_url}/tts/file",
                json={
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "prompt": prompt,
                    "return_timestamps": return_timestamps,
                },
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            native_sr = int(resp.headers.get("X-Native-Sample-Rate", "24000"))
            sentences = None
            sentences_raw = resp.headers.get("X-Sentences")
            if sentences_raw:
                try:
                    sentences = json.loads(sentences_raw)
                except json.JSONDecodeError:
                    sentences = None
            return resp.content, native_sr, sentences
        except httpx.HTTPError as exc:
            logger.exception("Qwen3-TTS /tts/file 调用失败 (%s)", base_url)
            raise DefaultServerErrorException(
                f"qwen3-tts service error: {exc}"
            ) from exc
        finally:
            self._pool.release(idx)

    def _get_voices_listing(self) -> Dict[str, Any]:
        idx, base_url = self._pool.acquire()
        try:
            r = _get_httpx_client().get(
                f"{base_url}/voices",
                headers=self._headers,
                timeout=self._timeout,
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            logger.warning("拉取 Qwen3-TTS 音色列表失败: %s", exc)
            return {"preset": [], "clone": [], "all": [], "registry": {}, "info": {}}
        finally:
            self._pool.release(idx)

    def _get_voice_info(self, name: str) -> Dict[str, Any]:
        idx, base_url = self._pool.acquire()
        try:
            r = _get_httpx_client().get(
                f"{base_url}/voices/{name}",
                headers=self._headers,
                timeout=self._timeout,
            )
            r.raise_for_status()
            return r.json()
        finally:
            self._pool.release(idx)

    def _post_voice(self, name: str, prompt_text: str, wav_path) -> Dict[str, Any]:
        with open(wav_path, "rb") as fp:
            files = {"audio": (wav_path.name, fp, "audio/wav")}
            data = {"name": name, "prompt_text": prompt_text}
            r = _get_httpx_client().post(
                f"{self._primary_url}/voices",
                files=files,
                data=data,
                headers=self._headers,
                timeout=self._timeout,
            )
        r.raise_for_status()
        result = r.json()
        self._broadcast_reload()
        return result

    def _delete_voice(self, name: str) -> Dict[str, Any]:
        r = _get_httpx_client().delete(
            f"{self._primary_url}/voices/{name}",
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        result = r.json()
        self._broadcast_reload()
        return result

    def _post_voices_refresh(self) -> Dict[str, Any]:
        r = _get_httpx_client().post(
            f"{self._primary_url}/voices/refresh",
            headers=self._headers,
            timeout=self._timeout,
        )
        r.raise_for_status()
        result = r.json()
        self._broadcast_reload()
        return result


def make_qwen3_tts_http_engine() -> Qwen3TTSHttpEngine:
    urls = _split_urls(settings.QWEN3_TTS_SERVICE_URLS)
    if not urls:
        raise DefaultServerErrorException(
            "QWEN3_TTS_SERVICE_URLS 未配置 — TTS_ENGINE=qwen3-tts 需要 "
            "services/qwen3_tts 子服务"
        )
    return Qwen3TTSHttpEngine(
        urls=urls,
        internal_token=settings.INTERNAL_SERVICE_TOKEN or "",
        timeout=settings.SERVICE_REQUEST_TIMEOUT,
    )


def make_qwen3_tts_vllm_omni_http_engine() -> Qwen3TTSHttpEngine:
    urls = _split_urls(settings.QWEN3_TTS_VLLM_OMNI_SERVICE_URLS)
    if not urls:
        raise DefaultServerErrorException(
            "QWEN3_TTS_VLLM_OMNI_SERVICE_URLS 未配置 — "
            "TTS_ENGINE=qwen3-tts-vllm-omni 需要 services/qwen3_tts_vllm_omni 子服务"
        )
    return Qwen3TTSHttpEngine(
        urls=urls,
        internal_token=settings.INTERNAL_SERVICE_TOKEN or "",
        timeout=settings.SERVICE_REQUEST_TIMEOUT,
    )
