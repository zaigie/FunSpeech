# -*- coding: utf-8 -*-
"""模型 / 子服务可达性预检

进程内模型预加载(VAD / PUNC / SFT / Clone)已迁到 services/* 子服务的
lifespan 中,网关只需校验 HTTP 客户端能与子服务握手即可。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _check_engine(engine_name: str, engine) -> Dict[str, Any]:
    record: Dict[str, Any] = {"loaded": False, "error": None, "device": ""}
    try:
        record["device"] = getattr(engine, "device", "")
        record["loaded"] = bool(engine.is_model_loaded())
        if not record["loaded"]:
            record["error"] = "is_model_loaded() == False"
            logger.warning("⚠️  %s 子服务未就绪: %s", engine_name, record["device"])
        else:
            logger.info("✅ %s 子服务就绪: %s", engine_name, record["device"])
    except Exception as exc:
        record["error"] = str(exc)
        logger.error("❌ %s 子服务连接失败: %s", engine_name, exc)
    return record


def preload_models() -> Dict[str, Any]:
    """预热 — 校验默认 ASR / TTS 子服务可达,但不阻塞启动失败的网关。"""
    from ..core.config import settings
    from ..services.asr.manager import get_model_manager
    from ..services.tts.engine import get_tts_engine

    result: Dict[str, Any] = {
        "asr_default_model": {"loaded": False, "model_id": None, "error": None},
        "asr_custom_models": {},
        "tts_engine": {"loaded": False, "error": None},
    }

    # ASR 默认模型(根据 models.json 中的 default 字段)
    try:
        manager = get_model_manager()
        config = manager.get_model_config()
        engine = manager.get_asr_engine()
        result["asr_default_model"] = {
            "loaded": engine.is_model_loaded(),
            "model_id": config.model_id,
            "error": None,
        }
        if result["asr_default_model"]["loaded"]:
            logger.info("✅ 默认 ASR(%s) 子服务就绪 [%s]", config.model_id, engine.device)
        else:
            logger.warning("⚠️  默认 ASR(%s) 子服务尚不可达", config.model_id)
    except Exception as exc:
        result["asr_default_model"]["error"] = str(exc)
        logger.error("❌ 默认 ASR 子服务校验失败: %s", exc)

    # 自定义 ASR(env: AUTO_LOAD_CUSTOM_ASR_MODELS=funasr-xxx,dolphin-xxx)
    auto_models_str = (settings.AUTO_LOAD_CUSTOM_ASR_MODELS or "").strip()
    if auto_models_str:
        for model_id in [m.strip() for m in auto_models_str.split(",") if m.strip()]:
            try:
                manager = get_model_manager()
                engine = manager.get_asr_engine(model_id)
                ok = engine.is_model_loaded()
                result["asr_custom_models"][model_id] = {
                    "loaded": ok,
                    "error": None if ok else "is_model_loaded() == False",
                }
                if ok:
                    logger.info("✅ 自定义 ASR(%s) 就绪", model_id)
                else:
                    logger.warning("⚠️  自定义 ASR(%s) 尚不可达", model_id)
            except Exception as exc:
                result["asr_custom_models"][model_id] = {
                    "loaded": False,
                    "error": str(exc),
                }
                logger.error("❌ 自定义 ASR(%s) 校验失败: %s", model_id, exc)

    # TTS
    try:
        tts_engine = get_tts_engine()
        result["tts_engine"] = _check_engine("TTS", tts_engine)
    except Exception as exc:
        result["tts_engine"]["error"] = str(exc)
        logger.error("❌ TTS 子服务校验失败: %s", exc)

    return result


def print_model_statistics(result: Dict[str, Any], use_logger: bool = True) -> None:
    """打印预热结果统计"""
    output = logger.info if use_logger else print

    output("=" * 60)
    output("📊 子服务连接统计:")
    output("-" * 60)

    if result["asr_default_model"]["loaded"]:
        output(
            "   ✅ 默认 ASR(%s)" % result["asr_default_model"]["model_id"]
        )
    elif result["asr_default_model"]["error"]:
        output("   ❌ 默认 ASR: %s" % result["asr_default_model"]["error"])

    for mid, status in result["asr_custom_models"].items():
        if status["loaded"]:
            output(f"   ✅ 自定义 ASR({mid})")
        elif status["error"]:
            output(f"   ❌ 自定义 ASR({mid}): {status['error']}")

    if result["tts_engine"]["loaded"]:
        output("   ✅ TTS")
    elif result["tts_engine"]["error"]:
        output("   ❌ TTS: %s" % result["tts_engine"]["error"])

    output("=" * 60)
