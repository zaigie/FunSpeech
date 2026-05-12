# -*- coding: utf-8 -*-
"""CosyVoice 子服务 HTTP 客户端引擎

实现与 CosyVoiceTTSEngine 公开方法兼容的 facade,内部走 HTTP 调
services/cosyvoice 子服务。网关 venv 不再需要 cosyvoice/torch 等重模型依赖,
但仍然负责:
  - 采样率重采样 (24kHz/22050Hz native -> 用户请求的目标采样率)
  - 音频格式转换 (WAV -> PCM/MP3 via app.utils.audio.save_audio_array)
  - 音量归一化
  - 文件落盘 (返回本地 temp 路径,与原引擎语义一致)
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
# 复用 asr 模块的 httpx 单例: 同一进程一份连接池, 不重复维护
from ..asr.http_engine import _get_httpx_client, get_async_httpx_client

logger = logging.getLogger(__name__)


def _split_urls(raw: str) -> List[str]:
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


class _HttpReplicaPool:
    def __init__(self, urls: List[str]):
        if not urls:
            raise ValueError("CosyVoiceHttpEngine requires at least one replica URL")
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


class _RemoteVoiceManager:
    """音色管理器 — HTTP 客户端版本

    提供与本地 VoiceManager 一致的公共方法签名,但实现全部走 HTTP。
    主要用户: app/services/tts/clone/* 的 API 路由 + 网关 TTS 引擎自身。
    """

    def __init__(self, engine: "CosyVoiceHttpEngine"):
        self._engine = engine
        self._cached_lists: Optional[Dict[str, Any]] = None
        self._cache_lock = threading.Lock()

    # 缓存策略: 首次拉,任何写操作后 invalidate
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
            logger.warning("add_voice 失败 %s: %s", voice_name, exc)
            return False

    def remove_voice(self, voice_name: str) -> bool:
        try:
            self._engine._delete_voice(voice_name)
            self._invalidate()
            return True
        except Exception as exc:
            logger.warning("remove_voice 失败 %s: %s", voice_name, exc)
            return False

    def refresh_voices(self) -> Tuple[int, int]:
        try:
            r = self._engine._post_voices_refresh()
            self._invalidate()
            return r.get("added", 0), r.get("total", 0)
        except Exception as exc:
            logger.warning("refresh_voices 失败: %s", exc)
            return 0, 0

    def add_all_voices(self) -> Tuple[int, int]:
        return self.refresh_voices()

    def get_registry_info(self) -> Dict[str, Any]:
        listing = self._fetch()
        return {
            "version": "2.0",
            "total_voices": len(listing.get("all", [])),
            "clone_voices": len(listing.get("clone", [])),
            "preset_voices": len(listing.get("preset", [])),
            "voices": listing.get("clone", []),
        }


class CosyVoiceHttpEngine:
    """HTTP 客户端引擎 — 子服务 services/cosyvoice 的网关侧 facade

    保留与原进程内 CosyVoiceTTSEngine 一致的公开方法签名(
    synthesize_speech / get_voices / voice_manager 等),网关代码无需感知
    底层是进程内还是 HTTP。流式合成走 iter_stream_audio_chunks (内部 WS)。

    多副本写副本约定:
      - URL 列表第一个 = 主写副本 (primary), 接收所有音色 CRUD
      - 写完成后向其它副本广播 POST /voices/reload, 让它们从磁盘热重载
        spk2info.pt 与 voice_registry.json
      - 读 / 合成请求仍走副本池调度
    """

    def __init__(
        self,
        urls: List[str],
        internal_token: str = "",
        timeout: float = 120.0,
    ):
        self._pool = _HttpReplicaPool(urls)
        self._urls = list(urls)
        self._primary_url = urls[0] if urls else ""
        self._timeout = timeout
        self._internal_token = internal_token
        self._headers: Dict[str, str] = {}
        if internal_token:
            self._headers["X-Internal-Token"] = internal_token
        self._voice_manager = _RemoteVoiceManager(self)
        self._sft_loaded = False
        self._clone_loaded = False
        self._cached_health_at = 0.0

    def _broadcast_reload(self) -> None:
        """向除 primary 外的所有副本广播 /voices/reload。

        失败仅打 warning,不抛异常 — 其它副本最坏只是看不到刚写的音色,
        重启或下次广播会同步。
        """
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
                        "广播 reload 到 %s 返回 %s: %s",
                        url, r.status_code, r.text[:200],
                    )
                else:
                    logger.debug("已广播 reload 到 %s", url)
            except httpx.HTTPError as exc:
                logger.warning("广播 reload 到 %s 失败: %s", url, exc)

    # ---------------------------------------------------------- 公共接口

    @property
    def voice_manager(self):
        return self._voice_manager

    @property
    def device(self) -> str:
        return f"remote:{','.join(self._pool._urls)}"

    def is_sft_model_loaded(self) -> bool:
        self._refresh_health_if_stale()
        return self._sft_loaded

    def is_clone_model_loaded(self) -> bool:
        self._refresh_health_if_stale()
        return self._clone_loaded

    def is_tts_model_loaded(self) -> bool:
        return self.is_sft_model_loaded() or self.is_clone_model_loaded()

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
        """整段合成 — 调子服务 /tts/file → 网关侧落盘 + 转码

        参数语义与 CosyVoiceTTSEngine.synthesize_speech 完全一致。
        """
        wav_bytes, native_sr, sentences = self._post_tts_file(
            text=text,
            voice=voice,
            speed=speed,
            prompt=prompt,
            return_timestamps=return_timestamps,
        )

        # 解出 WAV → numpy float32 (1, N) 与原引擎一致
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (1, N)

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
        if mode == "sft":
            return [v for v in listing.get("all", []) if v in listing.get("preset", [])]
        if mode == "clone":
            return listing.get("clone", [])
        return listing.get("all", [])

    def get_voices_info(self) -> Dict[str, Dict[str, Any]]:
        # 沿用 CosyVoiceTTSEngine 的 preset_info 结构, clone 走 registry
        preset_info = {
            "中文女": {"type": "preset", "language": "zh-CN", "gender": "female", "description": "标准中文女声"},
            "中文男": {"type": "preset", "language": "zh-CN", "gender": "male", "description": "标准中文男声"},
            "英文女": {"type": "preset", "language": "en-US", "gender": "female", "description": "标准英文女声"},
            "英文男": {"type": "preset", "language": "en-US", "gender": "male", "description": "标准英文男声"},
            "日语男": {"type": "preset", "language": "ja-JP", "gender": "male", "description": "标准日语男声"},
            "韩语女": {"type": "preset", "language": "ko-KR", "gender": "female", "description": "标准韩语女声"},
            "粤语女": {"type": "preset", "language": "zh-HK", "gender": "female", "description": "标准粤语女声"},
        }
        listing = self._get_voices_listing()
        registry_voices = listing.get("registry", {}) or {}

        out: Dict[str, Dict[str, Any]] = {}
        target_voices = self.get_voices()
        for voice in target_voices:
            if voice in preset_info:
                info = dict(preset_info[voice])
            elif voice in registry_voices:
                vi = registry_voices.get(voice) or {}
                info = {
                    "type": "clone",
                    "language": "zh-CN",
                    "gender": "unknown",
                    "description": f"零样本克隆音色：{voice}",
                    "reference_text": vi.get("reference_text", ""),
                    "audio_file": vi.get("audio_file", ""),
                    "added_at": vi.get("added_at", ""),
                }
            else:
                info = {
                    "type": "custom",
                    "language": "unknown",
                    "gender": "unknown",
                    "description": f"自定义音色：{voice}",
                }
            info.update({"name": voice, "sample_rate": 22050, "available": True})
            out[voice] = info
        return out

    def refresh_voices(self) -> None:
        # invalidate 缓存即可 — 子服务侧已经实时反映 registry
        self._voice_manager._invalidate()

    # ---------------------------------------------------------- 内部 HTTP

    def _refresh_health_if_stale(self) -> None:
        """刷新健康状态缓存。直接探 _urls,不走副本池 acquire/release —
        否则高频健康检查会把"活跃连接数"计数推高,污染最少连接调度。"""
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
            native_sr = int(resp.headers.get("X-Native-Sample-Rate", "22050"))
            sentences = None
            sentences_raw = resp.headers.get("X-Sentences")
            if sentences_raw:
                try:
                    sentences = json.loads(sentences_raw)
                except json.JSONDecodeError:
                    sentences = None
            return resp.content, native_sr, sentences
        except httpx.HTTPError as exc:
            logger.exception("CosyVoice /tts/file 调用失败 (%s)", base_url)
            raise DefaultServerErrorException(
                f"cosyvoice service error: {exc}"
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
            logger.warning("拉取音色列表失败: %s", exc)
            return {"preset": [], "clone": [], "all": [], "registry": {}}
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
        # 写操作必须打到 primary 副本, 否则 spk2info.pt 会冲突
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


    # ---------------------------------------------------------- WS 流式

    async def iter_stream_audio_chunks(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        prompt: str = "",
    ):
        """异步生成器: 流式合成,逐 chunk yield (audio_array, native_sr)

        内部打开一个内部 WS 到子服务 /tts/stream;子服务侧推 float32 PCM 块,
        本方法解码后逐 chunk 返回。网关侧(websocket_tts.py)按需做重采样和格式转换。
        """
        idx, base_url = self._pool.acquire()
        ws_url = base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        ) + "/tts/stream"

        if self._internal_token:
            sep = "&" if "?" in ws_url else "?"
            ws_url = f"{ws_url}{sep}token={self._internal_token}"

        # 用 websockets.asyncio.client 异步实现, 不阻塞事件循环
        # 每个 recv 都包 asyncio.wait_for, 上游卡死时不会让客户端永久 hang
        import asyncio as _asyncio

        from websockets.asyncio.client import connect

        try:
            async with connect(ws_url, open_timeout=self._timeout) as ws:
                await ws.send(
                    json.dumps(
                        {
                            "text": text,
                            "voice": voice,
                            "speed": speed,
                            "prompt": prompt,
                        }
                    )
                )
                # 第 1 帧 JSON: started + sample_rate
                try:
                    first = await _asyncio.wait_for(ws.recv(), timeout=self._timeout)
                except _asyncio.TimeoutError as exc:
                    raise DefaultServerErrorException(
                        "cosyvoice ws started 帧超时"
                    ) from exc
                if isinstance(first, (bytes, bytearray)):
                    raise DefaultServerErrorException(
                        "cosyvoice ws started 帧非文本"
                    )
                meta = json.loads(first)
                if meta.get("type") == "error":
                    raise DefaultServerErrorException(
                        f"cosyvoice stream error: {meta.get('message')}"
                    )
                if meta.get("type") != "started":
                    raise DefaultServerErrorException(
                        f"cosyvoice stream unexpected first frame: {meta}"
                    )
                native_sr = int(meta.get("sample_rate", 24000))

                while True:
                    try:
                        msg = await _asyncio.wait_for(
                            ws.recv(), timeout=self._timeout
                        )
                    except _asyncio.TimeoutError as exc:
                        raise DefaultServerErrorException(
                            "cosyvoice stream recv 超时"
                        ) from exc
                    if isinstance(msg, (bytes, bytearray)):
                        chunk = np.frombuffer(msg, dtype=np.float32)
                        if chunk.size == 0:
                            continue
                        # 与本地 inference_sft 输出 shape 一致: (1, N)
                        yield chunk.reshape(1, -1), native_sr
                    else:
                        evt = json.loads(msg)
                        if evt.get("type") == "done":
                            return
                        if evt.get("type") == "error":
                            raise DefaultServerErrorException(
                                f"cosyvoice stream error: {evt.get('message')}"
                            )
        finally:
            self._pool.release(idx)


def make_cosyvoice_http_engine() -> CosyVoiceHttpEngine:
    urls = _split_urls(settings.COSYVOICE_SERVICE_URLS)
    if not urls:
        raise DefaultServerErrorException(
            "COSYVOICE_SERVICE_URLS 未配置 — 网关需要通过 services/cosyvoice "
            "子服务才能使用 TTS"
        )
    return CosyVoiceHttpEngine(
        urls=urls,
        internal_token=settings.INTERNAL_SERVICE_TOKEN or "",
        timeout=settings.SERVICE_REQUEST_TIMEOUT,
    )
