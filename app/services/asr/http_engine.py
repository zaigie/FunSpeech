# -*- coding: utf-8 -*-
"""ASR 子服务 HTTP 客户端引擎

实现 BaseASREngine / RealTimeASREngine 的 HTTP 客户端版本，
让网关无需 import funasr/dolphin/torch 等模型库，全部通过 HTTP
调用 services/* 下的子服务。

当前(Step 3)只支持 transcribe_file。WS 流式在 Step 4 加上。
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Dict, List, Optional, Tuple

import httpx

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


class _BaseHttpAsrEngine(BaseASREngine):
    """子服务 HTTP 客户端引擎基类"""

    def __init__(self, urls: List[str], internal_token: str = "", timeout: float = 60.0):
        self._pool = _HttpReplicaPool(urls)
        self._timeout = timeout
        self._headers = {}
        if internal_token:
            self._headers["X-Internal-Token"] = internal_token

    @property
    def supports_realtime(self) -> bool:
        return False

    @property
    def device(self) -> str:
        return f"remote:{','.join(self._pool._urls)}"

    def is_model_loaded(self) -> bool:
        # 至少一个副本健康即视为已加载
        for url in self._pool._urls:
            try:
                r = httpx.get(f"{url}/health", timeout=5.0, headers=self._headers)
                if r.status_code == 200:
                    return True
            except httpx.HTTPError:
                continue
        return False


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
        self._headers: Dict[str, str] = {}
        if internal_token:
            self._headers["X-Internal-Token"] = internal_token

    @property
    def supports_realtime(self) -> bool:
        # FunASR 子服务支持流式,但 Step 3 暂未接入
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
        # Step 4 实装。当前切换到 HTTP 模式时,WS 流式会回退到旧的进程内引擎,
        # 详见 manager._create_engine 的旗标分支。
        raise NotImplementedError(
            "FunASRHttpEngine.transcribe_websocket 将在 Step 4 实装；"
            "当前请保持 USE_FUNASR_SERVICE=false 以走进程内引擎处理流式请求。"
        )


def make_funasr_http_engine() -> FunASRHttpEngine:
    """从 settings 构造 FunASRHttpEngine"""
    urls = _split_urls(settings.FUNASR_SERVICE_URLS)
    if not urls:
        raise DefaultServerErrorException(
            "USE_FUNASR_SERVICE 已启用,但未配置 FUNASR_SERVICE_URLS"
        )
    return FunASRHttpEngine(
        urls=urls,
        internal_token=settings.INTERNAL_SERVICE_TOKEN or "",
        timeout=settings.SERVICE_REQUEST_TIMEOUT,
    )
