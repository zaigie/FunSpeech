# -*- coding: utf-8 -*-
"""TTS 引擎入口。

默认走 services/cosyvoice 子服务;设置 TTS_ENGINE=qwen3-tts 时走本地
services/qwen3_tts 子服务。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from .http_engine import make_cosyvoice_http_engine

logger = logging.getLogger(__name__)


# 全局 TTS 引擎实例
_tts_engine: Optional[Any] = None
_tts_engine_lock = threading.Lock()

_TTS_ENGINE_ALIASES = {
    "cosyvoice": "cosyvoice",
    "cosy": "cosyvoice",
    "qwen": "qwen3-tts",
    "qwen-tts": "qwen3-tts",
    "qwentts": "qwen3-tts",
    "qwen3": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
}


def normalize_tts_engine(value: str) -> str:
    engine_name = (value or "cosyvoice").strip().lower()
    try:
        return _TTS_ENGINE_ALIASES[engine_name]
    except KeyError as exc:
        raise DefaultServerErrorException(
            f"unsupported TTS_ENGINE={value!r}; expected cosyvoice or qwen3-tts"
        ) from exc


def get_tts_engine() -> Any:
    """获取全局 TTS 引擎。"""
    global _tts_engine

    with _tts_engine_lock:
        if _tts_engine is None:
            engine_name = normalize_tts_engine(settings.TTS_ENGINE)
            if engine_name == "qwen3-tts":
                from .qwen3_http_engine import make_qwen3_tts_http_engine

                _tts_engine = make_qwen3_tts_http_engine()
            else:
                _tts_engine = make_cosyvoice_http_engine()
            logger.info("TTS engine initialized: %s", engine_name)
        return _tts_engine
