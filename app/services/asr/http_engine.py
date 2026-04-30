# -*- coding: utf-8 -*-
"""ASR 子服务 HTTP 客户端引擎

实现 BaseASREngine / RealTimeASREngine 的 HTTP 客户端版本,
让网关无需 import funasr/dolphin/torch 等模型库,全部通过 HTTP
调用 services/* 下的子服务。

包含:
  - transcribe_file: POST /asr/file
  - 流式 ASR: 持有内部 WS 会话, cache 状态在子服务侧
  - punc_offline: POST /asr/punc (供网关在 SentenceEnd 时给整句打标点)
"""

from __future__ import annotations

import json
import logging
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from .engine import BaseASREngine, RealTimeASREngine

logger = logging.getLogger(__name__)


def _split_urls(raw: str) -> List[str]:
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


class _HttpReplicaPool:
    """简易副本池: 最少连接 + 健康过滤(被动: 失败一次即记一次,后续依赖 manager 的健康检查升级)"""

    def __init__(self, urls: List[str]):
        if not urls:
            raise ValueError("HTTP engine requires at least one replica URL")
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


class _RealtimeASRSession:
    """单个外部 WS 连接对应的内部 WS 会话。

    线程安全;send_chunk / flush 同步阻塞,适合在 run_sync 线程池里调用。
    通过持有一个长连 websocket 给子服务,cache 状态保留在子服务端。
    """

    def __init__(
        self,
        ws_url: str,
        params: Dict[str, Any],
        internal_token: str = "",
        timeout: float = 60.0,
    ):
        # 延迟 import 避免主项目环境无 websockets.sync 时启动失败
        from websockets.sync.client import connect as ws_connect

        full_url = ws_url
        if internal_token:
            sep = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{sep}token={internal_token}"

        self._ws = ws_connect(full_url, open_timeout=timeout, close_timeout=5.0)
        self._lock = threading.Lock()
        self._closed = False

        # 发送 start 帧
        start = {"op": "start", **params}
        self._ws.send(json.dumps(start))
        ack = json.loads(self._ws.recv(timeout=timeout))
        if ack.get("type") != "started":
            raise DefaultServerErrorException(
                f"funasr stream start failed: {ack}"
            )

    def send_chunk(self, audio_array_float32: np.ndarray) -> Dict[str, Any]:
        """送一个 PCM chunk(float32 [-1,1] 范围),返回 {text, text_punc, is_silence}"""
        with self._lock:
            if self._closed:
                return {"text": "", "text_punc": "", "is_silence": True}

            pcm_int16 = (
                np.asarray(audio_array_float32, dtype=np.float32) * 32768.0
            ).astype(np.int16)
            self._ws.send(pcm_int16.tobytes())

            # 读直到收到 partial 或 error
            while True:
                raw = self._ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    continue  # 子服务不应发二进制,忽略
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "partial":
                    return {
                        "text": msg.get("text", ""),
                        "text_punc": msg.get("text_punc", ""),
                        "is_silence": bool(msg.get("is_silence", False)),
                    }
                if kind == "error":
                    raise DefaultServerErrorException(
                        f"funasr stream error: {msg.get('message')}"
                    )
                # started 等其它消息忽略

    def flush(self) -> str:
        with self._lock:
            if self._closed:
                return ""
            self._ws.send(json.dumps({"op": "flush"}))
            while True:
                raw = self._ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "flushed":
                    return msg.get("text", "")
                if msg.get("type") == "error":
                    raise DefaultServerErrorException(
                        f"funasr stream flush error: {msg.get('message')}"
                    )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._ws.send(json.dumps({"op": "close"}))
            except Exception:
                pass
            try:
                self._ws.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _HttpRealtimeModel:
    """funasr 风格的伪 realtime_model

    现有 websocket_asr.py 调 asr_engine.realtime_model.generate(input, cache,
    is_final, chunk_size=...) 期望返回列表 [{"text": "..."}]。我们这里把
    cache 当做"携带 session 句柄的 dict"用,首次调用懒开 session,后续复用。
    """

    _SESSION_KEY = "__funasr_http_session__"

    def __init__(self, engine: "FunASRHttpEngine"):
        self._engine = engine

    def generate(self, *, input, cache, is_final, chunk_size=None, **kwargs):
        params = {
            "chunk_size": chunk_size or [0, 10, 5],
            "encoder_lookback": kwargs.get("encoder_chunk_look_back", 4),
            "decoder_lookback": kwargs.get("decoder_chunk_look_back", 1),
            "sample_rate": int(kwargs.get("sample_rate", 16000)),
            "format": kwargs.get("format", "pcm"),
            "enable_realtime_punc": settings.ASR_ENABLE_REALTIME_PUNC,
        }

        session = cache.get(self._SESSION_KEY)
        if session is None:
            session = self._engine._open_realtime_session(params)
            cache[self._SESSION_KEY] = session

        if is_final:
            text = session.flush()
            # flush 后子服务侧已 cache.clear, 我们也丢掉 session 让 cache 重置干净
            try:
                session.close()
            except Exception:
                pass
            cache.pop(self._SESSION_KEY, None)
            return [{"text": text}] if text else []

        # 非 final: input 是 numpy 数组
        result = session.send_chunk(np.asarray(input, dtype=np.float32))
        return [
            {
                "text": result["text"],
                "_text_punc": result["text_punc"],
                "_is_silence": result["is_silence"],
            }
        ]


class FunASRHttpEngine(RealTimeASREngine):
    """FunASR 子服务 HTTP 客户端

    transcribe_file: POST /asr/file
    transcribe_websocket: 见 Step 4 — 当前抛 NotImplementedError
    """

    def __init__(
        self,
        urls: List[str],
        internal_token: str = "",
        timeout: float = 60.0,
    ):
        if not urls:
            raise ValueError("FunASRHttpEngine requires at least one replica URL")
        self._pool = _HttpReplicaPool(urls)
        self._timeout = timeout
        self._internal_token = internal_token
        self._headers: Dict[str, str] = {}
        if internal_token:
            self._headers["X-Internal-Token"] = internal_token

        # 网关 websocket_asr.py 直接访问 .realtime_model.generate(...);
        # 我们提供一个伪对象转发到内部 WS。
        self.realtime_model = _HttpRealtimeModel(self)

    def _open_realtime_session(self, params: Dict[str, Any]) -> _RealtimeASRSession:
        # 选副本 — 注意 session 是长连,这里只占一次 acquire/release
        idx, base_url = self._pool.acquire()
        ws_url = base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        ) + "/asr/stream/v1"
        try:
            session = _RealtimeASRSession(
                ws_url=ws_url,
                params=params,
                internal_token=self._internal_token,
                timeout=self._timeout,
            )
        except Exception:
            self._pool.release(idx)
            raise

        # session 关闭时释放副本
        original_close = session.close

        def close_and_release():
            try:
                original_close()
            finally:
                self._pool.release(idx)

        session.close = close_and_release  # type: ignore[method-assign]
        return session

    def punc_offline(self, text: str) -> str:
        """供网关在 SentenceEnd 时给整句打离线标点。"""
        if not text:
            return text
        idx, base_url = self._pool.acquire()
        try:
            resp = httpx.post(
                f"{base_url}/asr/punc",
                json={"text": text, "mode": "offline"},
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("text", text)
        except httpx.HTTPError as exc:
            logger.warning("funasr /asr/punc 调用失败: %s", exc)
            return text
        finally:
            self._pool.release(idx)

    @property
    def supports_realtime(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return f"remote:{','.join(self._pool._urls)}"

    def is_model_loaded(self) -> bool:
        for url in self._pool._urls:
            try:
                r = httpx.get(f"{url}/health", timeout=5.0, headers=self._headers)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
        return False

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        dolphin_lang_sym: str = "zh",
        dolphin_region_sym: str = "SHANGHAI",
    ) -> str:
        idx, base_url = self._pool.acquire()
        try:
            with open(audio_path, "rb") as fp:
                files = {"audio": (audio_path.rsplit("/", 1)[-1], fp, "application/octet-stream")}
                data = {
                    "hotwords": hotwords,
                    "enable_punctuation": str(enable_punctuation).lower(),
                    "enable_itn": str(enable_itn).lower(),
                    "enable_vad": str(enable_vad).lower(),
                    "sample_rate": str(sample_rate),
                }
                resp = httpx.post(
                    f"{base_url}/asr/file",
                    files=files,
                    data=data,
                    headers=self._headers,
                    timeout=self._timeout,
                )
            resp.raise_for_status()
            return resp.json().get("text", "")
        except httpx.HTTPError as exc:
            logger.exception("FunASR HTTP 调用失败 (%s)", base_url)
            raise DefaultServerErrorException(f"funasr service error: {exc}") from exc
        finally:
            self._pool.release(idx)

    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        # 网关 websocket_asr.py 实际走 self.realtime_model.generate(...) 路径,
        # 不调本方法。保留实现以便其它代码路径直接调用。
        if cache is None:
            cache = {}
        sample_rate = int(kwargs.get("sample_rate", 16000))
        if audio_chunk:
            audio_array = (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )
        else:
            audio_array = np.array([], dtype=np.float32)
        result = self.realtime_model.generate(
            input=audio_array,
            cache=cache,
            is_final=is_final,
            chunk_size=kwargs.get("chunk_size", [0, 10, 5]),
            sample_rate=sample_rate,
            **{k: v for k, v in kwargs.items() if k not in ("chunk_size", "sample_rate")},
        )
        if result and len(result):
            return result[0].get("text", "")
        return ""


def make_funasr_http_engine() -> FunASRHttpEngine:
    """从 settings 构造 FunASRHttpEngine"""
    urls = _split_urls(settings.FUNASR_SERVICE_URLS)
    if not urls:
        raise DefaultServerErrorException(
            "FUNASR_SERVICE_URLS 未配置 — 网关需要通过 services/funasr 子服务"
            "才能使用 funasr 引擎"
        )
    return FunASRHttpEngine(
        urls=urls,
        internal_token=settings.INTERNAL_SERVICE_TOKEN or "",
        timeout=settings.SERVICE_REQUEST_TIMEOUT,
    )


class DolphinHttpEngine(BaseASREngine):
    """Dolphin 子服务 HTTP 客户端 — 仅离线"""

    def __init__(
        self,
        urls: List[str],
        internal_token: str = "",
        timeout: float = 60.0,
    ):
        if not urls:
            raise ValueError("DolphinHttpEngine requires at least one replica URL")
        self._pool = _HttpReplicaPool(urls)
        self._timeout = timeout
        self._headers: Dict[str, str] = {}
        if internal_token:
            self._headers["X-Internal-Token"] = internal_token

    @property
    def supports_realtime(self) -> bool:
        return False

    @property
    def device(self) -> str:
        return f"remote:{','.join(self._pool._urls)}"

    def is_model_loaded(self) -> bool:
        for url in self._pool._urls:
            try:
                r = httpx.get(f"{url}/health", timeout=5.0, headers=self._headers)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
        return False

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        dolphin_lang_sym: str = "zh",
        dolphin_region_sym: str = "SHANGHAI",
    ) -> str:
        idx, base_url = self._pool.acquire()
        try:
            with open(audio_path, "rb") as fp:
                files = {
                    "audio": (
                        audio_path.rsplit("/", 1)[-1],
                        fp,
                        "application/octet-stream",
                    )
                }
                data = {
                    "lang_sym": dolphin_lang_sym,
                    "region_sym": dolphin_region_sym,
                    "sample_rate": str(sample_rate),
                }
                resp = httpx.post(
                    f"{base_url}/asr/file",
                    files=files,
                    data=data,
                    headers=self._headers,
                    timeout=self._timeout,
                )
            resp.raise_for_status()
            text = resp.json().get("text", "")
        except httpx.HTTPError as exc:
            logger.exception("Dolphin HTTP 调用失败 (%s)", base_url)
            raise DefaultServerErrorException(f"dolphin service error: {exc}") from exc
        finally:
            self._pool.release(idx)

        # 标点 / ITN: dolphin 子服务不做, 委托给 funasr 子服务
        if enable_punctuation and text:
            text = self._punctuate_via_funasr(text)
        if enable_itn and text:
            from ...utils.text_processing import apply_itn_to_text

            text = apply_itn_to_text(text)
        return text

    def _punctuate_via_funasr(self, text: str) -> str:
        """复用 funasr 子服务的 /asr/punc 给 dolphin 输出加标点"""
        urls = _split_urls(settings.FUNASR_SERVICE_URLS)
        if not urls:
            logger.debug(
                "dolphin 启用 punctuation 但未配置 FUNASR_SERVICE_URLS,跳过"
            )
            return text

        funasr_url = urls[0]
        headers = {}
        if settings.INTERNAL_SERVICE_TOKEN:
            headers["X-Internal-Token"] = settings.INTERNAL_SERVICE_TOKEN
        try:
            resp = httpx.post(
                f"{funasr_url}/asr/punc",
                json={"text": text, "mode": "offline"},
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("text", text)
        except httpx.HTTPError as exc:
            logger.warning("dolphin 借 funasr 加标点失败: %s", exc)
            return text


def make_dolphin_http_engine() -> DolphinHttpEngine:
    urls = _split_urls(settings.DOLPHIN_SERVICE_URLS)
    if not urls:
        raise DefaultServerErrorException(
            "DOLPHIN_SERVICE_URLS 未配置 — 网关需要通过 services/dolphin 子服务"
            "才能使用 dolphin 引擎"
        )
    return DolphinHttpEngine(
        urls=urls,
        internal_token=settings.INTERNAL_SERVICE_TOKEN or "",
        timeout=settings.SERVICE_REQUEST_TIMEOUT,
    )


class _Qwen3RealtimeASRSession(_RealtimeASRSession):
    """qwen3-asr 子服务的内部 WS 会话

    协议字段名与 funasr 兼容(text / text_punc / is_silence),
    qwen3 不产 text_punc(模型自带标点)。
    """


class _Qwen3HttpRealtimeModel:
    """qwen3-asr 用的伪 funasr.realtime_model"""

    _SESSION_KEY = "__qwen3_asr_http_session__"

    def __init__(self, engine: "Qwen3AsrVllmHttpEngine"):
        self._engine = engine

    def generate(self, *, input, cache, is_final, chunk_size=None, **kwargs):
        params = {
            "sample_rate": int(kwargs.get("sample_rate", 16000)),
            "format": kwargs.get("format", "pcm"),
            "unfixed_chunk_num": int(kwargs.get("unfixed_chunk_num", 2)),
            "unfixed_token_num": int(kwargs.get("unfixed_token_num", 5)),
            "chunk_size_sec": float(kwargs.get("chunk_size_sec", 2.0)),
        }

        session = cache.get(self._SESSION_KEY)
        if session is None:
            session = self._engine._open_realtime_session(params)
            cache[self._SESSION_KEY] = session

        if is_final:
            text = session.flush()
            try:
                session.close()
            except Exception:
                pass
            cache.pop(self._SESSION_KEY, None)
            return [{"text": text}] if text else []

        result = session.send_chunk(np.asarray(input, dtype=np.float32))
        return [
            {
                "text": result["text"],
                "_text_punc": result.get("text_punc", ""),
                "_is_silence": result["is_silence"],
            }
        ]


class Qwen3AsrVllmHttpEngine(RealTimeASREngine):
    """Qwen3-ASR vLLM 子服务 HTTP 客户端"""

    def __init__(
        self,
        urls: List[str],
        internal_token: str = "",
        timeout: float = 60.0,
    ):
        if not urls:
            raise ValueError("Qwen3AsrVllmHttpEngine requires at least one replica URL")
        self._pool = _HttpReplicaPool(urls)
        self._timeout = timeout
        self._internal_token = internal_token
        self._headers: Dict[str, str] = {}
        if internal_token:
            self._headers["X-Internal-Token"] = internal_token

        self.realtime_model = _Qwen3HttpRealtimeModel(self)

    def _open_realtime_session(self, params: Dict[str, Any]) -> _Qwen3RealtimeASRSession:
        idx, base_url = self._pool.acquire()
        ws_url = base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        ) + "/asr/stream/v1"
        try:
            session = _Qwen3RealtimeASRSession(
                ws_url=ws_url,
                params=params,
                internal_token=self._internal_token,
                timeout=self._timeout,
            )
        except Exception:
            self._pool.release(idx)
            raise

        original_close = session.close

        def close_and_release():
            try:
                original_close()
            finally:
                self._pool.release(idx)

        session.close = close_and_release  # type: ignore[method-assign]
        return session

    @property
    def supports_realtime(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return f"remote:{','.join(self._pool._urls)}"

    def is_model_loaded(self) -> bool:
        for url in self._pool._urls:
            try:
                r = httpx.get(f"{url}/health", timeout=5.0, headers=self._headers)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
        return False

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,  # qwen3 模型自带标点,此参数忽略
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        dolphin_lang_sym: str = "zh",
        dolphin_region_sym: str = "SHANGHAI",
    ) -> str:
        idx, base_url = self._pool.acquire()
        try:
            with open(audio_path, "rb") as fp:
                files = {
                    "audio": (
                        audio_path.rsplit("/", 1)[-1],
                        fp,
                        "application/octet-stream",
                    )
                }
                data = {
                    "language": dolphin_lang_sym or "",  # 复用现有字段
                    "sample_rate": str(sample_rate),
                }
                resp = httpx.post(
                    f"{base_url}/asr/file",
                    files=files,
                    data=data,
                    headers=self._headers,
                    timeout=self._timeout,
                )
            resp.raise_for_status()
            text = resp.json().get("text", "")
        except httpx.HTTPError as exc:
            logger.exception("Qwen3-ASR HTTP 调用失败 (%s)", base_url)
            raise DefaultServerErrorException(f"qwen3-asr service error: {exc}") from exc
        finally:
            self._pool.release(idx)

        if enable_itn and text:
            from ...utils.text_processing import apply_itn_to_text

            text = apply_itn_to_text(text)
        return text

    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        if cache is None:
            cache = {}
        if audio_chunk:
            audio_array = (
                np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            )
        else:
            audio_array = np.array([], dtype=np.float32)
        result = self.realtime_model.generate(
            input=audio_array,
            cache=cache,
            is_final=is_final,
            **kwargs,
        )
        if result and len(result):
            return result[0].get("text", "")
        return ""


def make_qwen3_asr_http_engine() -> Qwen3AsrVllmHttpEngine:
    urls = _split_urls(settings.QWEN3_ASR_SERVICE_URLS)
    if not urls:
        raise DefaultServerErrorException(
            "QWEN3_ASR_SERVICE_URLS 未配置 — 网关需要通过 "
            "services/qwen3_asr_vllm 子服务才能使用 qwen3-asr 引擎"
        )
    return Qwen3AsrVllmHttpEngine(
        urls=urls,
        internal_token=settings.INTERNAL_SERVICE_TOKEN or "",
        timeout=settings.SERVICE_REQUEST_TIMEOUT,
    )
