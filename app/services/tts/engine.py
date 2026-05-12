# -*- coding: utf-8 -*-
"""TTS 引擎入口

进程内 CosyVoiceTTSEngine / MultiGPUTTSEngine 已删除,所有合成走
services/cosyvoice 子服务的 HTTP 客户端 (CosyVoiceHttpEngine)。
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .http_engine import CosyVoiceHttpEngine, make_cosyvoice_http_engine

logger = logging.getLogger(__name__)


# 全局 TTS 引擎实例
_tts_engine: Optional[CosyVoiceHttpEngine] = None
_tts_engine_lock = threading.Lock()


def get_tts_engine() -> CosyVoiceHttpEngine:
    """获取全局 TTS 引擎(CosyVoice 子服务 HTTP 客户端)"""
    global _tts_engine

    with _tts_engine_lock:
        if _tts_engine is None:
            _tts_engine = make_cosyvoice_http_engine()
        return _tts_engine
