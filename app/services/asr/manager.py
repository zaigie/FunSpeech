# -*- coding: utf-8 -*-
"""ASR 模型管理器

读取 models.json 决定每个 model_id 用哪个引擎(funasr / dolphin / qwen3-asr),
所有引擎一律走对应子服务的 HTTP 客户端。进程内推理已不再支持。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException, InvalidParameterException
from .engine import BaseASREngine

logger = logging.getLogger(__name__)


class ModelConfig:
    """模型配置类"""

    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.name = config["name"]
        self.engine = config["engine"]
        self.description = config.get("description", "")
        self.languages = config.get("languages", [])
        self.is_default = config.get("default", False)
        self.size = config.get("size")
        self.supports_realtime = config.get("supports_realtime", False)

        self.models = config.get("models", {})
        self.offline_model_path = self.models.get("offline")
        self.realtime_model_path = self.models.get("realtime")

    @property
    def has_offline_model(self) -> bool:
        return bool(self.offline_model_path)

    @property
    def has_realtime_model(self) -> bool:
        return bool(self.realtime_model_path)

    def get_model_path(self, model_type: str = "offline") -> Optional[str]:
        if model_type == "offline":
            return self.offline_model_path
        if model_type == "realtime":
            return self.realtime_model_path
        return None


class ModelManager:
    """ASR 模型管理器,支持多模型缓存"""

    def __init__(self):
        self._models_config: Dict[str, ModelConfig] = {}
        self._loaded_engines: Dict[str, BaseASREngine] = {}
        self._default_model_id: Optional[str] = None
        self._load_models_config()

    def _load_models_config(self) -> None:
        models_file = Path(settings.models_config_path)
        if not models_file.exists():
            raise DefaultServerErrorException("models.json 配置文件不存在")

        try:
            with open(models_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            for model_id, model_config in config["models"].items():
                self._models_config[model_id] = ModelConfig(model_id, model_config)
                if model_config.get("default", False):
                    self._default_model_id = model_id

            if not self._default_model_id and self._models_config:
                self._default_model_id = list(self._models_config.keys())[0]

        except (json.JSONDecodeError, KeyError) as exc:
            raise DefaultServerErrorException(
                f"模型配置文件格式错误: {exc}"
            )

    def get_model_config(self, model_id: Optional[str] = None) -> ModelConfig:
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        if model_id not in self._models_config:
            available = ", ".join(self._models_config.keys())
            raise InvalidParameterException(
                f"未知的模型: {model_id},可用模型: {available}"
            )
        return self._models_config[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        for model_id, config in self._models_config.items():
            loaded = model_id in self._loaded_engines
            models.append(
                {
                    "id": model_id,
                    "name": config.name,
                    "engine": config.engine,
                    "description": config.description,
                    "languages": config.languages,
                    "default": config.is_default,
                    "loaded": loaded,
                    "supports_realtime": config.supports_realtime,
                    "offline_model": (
                        {"path": config.offline_model_path, "exists": True}
                        if config.offline_model_path
                        else None
                    ),
                    "realtime_model": (
                        {"path": config.realtime_model_path, "exists": True}
                        if config.realtime_model_path
                        else None
                    ),
                    "asr_model_mode": settings.ASR_MODEL_MODE,
                }
            )
        return models

    def get_asr_engine(self, model_id: Optional[str] = None) -> BaseASREngine:
        """获取 ASR 引擎(全部走子服务 HTTP 客户端)"""
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        if model_id in self._loaded_engines:
            return self._loaded_engines[model_id]

        config = self.get_model_config(model_id)
        engine = self._create_engine(config)
        self._loaded_engines[model_id] = engine
        return engine

    def _create_engine(self, config: ModelConfig) -> BaseASREngine:
        engine_type = config.engine.lower()

        if engine_type == "funasr":
            from .http_engine import make_funasr_http_engine

            logger.info(
                "engine=funasr -> services/funasr (urls=%s)",
                settings.FUNASR_SERVICE_URLS,
            )
            return make_funasr_http_engine()

        if engine_type == "dolphin":
            from .http_engine import make_dolphin_http_engine

            logger.info(
                "engine=dolphin -> services/dolphin (urls=%s)",
                settings.DOLPHIN_SERVICE_URLS,
            )
            return make_dolphin_http_engine()

        if engine_type == "qwen3-asr":
            from .http_engine import make_qwen3_asr_http_engine

            logger.info(
                "engine=qwen3-asr -> services/qwen3_asr_vllm (urls=%s)",
                settings.QWEN3_ASR_SERVICE_URLS,
            )
            return make_qwen3_asr_http_engine()

        raise InvalidParameterException(f"不支持的引擎类型: {config.engine}")

    def unload_model(self, model_id: str) -> bool:
        """卸载缓存(实际模型在子服务里,这里只是清本地引用)"""
        if model_id in self._loaded_engines:
            del self._loaded_engines[model_id]
            return True
        return False

    def get_memory_usage(self) -> Dict[str, Any]:
        return {
            "model_list": list(self._loaded_engines.keys()),
            "loaded_count": len(self._loaded_engines),
            "asr_model_mode": settings.ASR_MODEL_MODE,
            "note": "模型在 services/* 子服务进程内, 本进程仅为 HTTP 客户端",
        }

    def clear_cache(self) -> None:
        self._loaded_engines.clear()

    def validate_model_mode_compatibility(self, model_id: str) -> Dict[str, Any]:
        config = self.get_model_config(model_id)
        mode = settings.ASR_MODEL_MODE.lower()
        errors: List[str] = []

        if mode == "offline" and not config.has_offline_model:
            errors.append(
                f"模型 {model_id} 没有离线版本,但 ASR_MODEL_MODE 设置为 offline"
            )
        elif mode == "realtime" and not config.has_realtime_model:
            errors.append(
                f"模型 {model_id} 没有实时版本,但 ASR_MODEL_MODE 设置为 realtime"
            )
        elif mode == "all" and not (
            config.has_offline_model or config.has_realtime_model
        ):
            errors.append(f"模型 {model_id} 既没有离线版本也没有实时版本")

        return {
            "model_id": model_id,
            "mode": mode,
            "errors": errors,
            "compatible": not errors,
        }


# 全局模型管理器实例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
