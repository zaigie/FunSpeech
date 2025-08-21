# -*- coding: utf-8 -*-
"""
ASR模型管理器
支持多模型缓存和动态加载
"""

import json
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException, InvalidParameterException
from .engine import ASREngine, FunASREngine, DolphinEngine


class ModelConfig:
    """模型配置类"""

    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.name = config["name"]
        self.path = config["path"]
        self.engine = config["engine"]
        self.description = config.get("description", "")
        self.languages = config.get("languages", [])
        self.is_default = config.get("default", False)
        self.size = config.get("size")  # 用于dolphin模型


class ModelManager:
    """模型管理器，支持多模型缓存"""

    def __init__(self):
        self._models_config: Dict[str, ModelConfig] = {}
        self._loaded_engines: Dict[str, ASREngine] = {}
        self._default_model_id: Optional[str] = None
        self._load_models_config()

    def _load_models_config(self) -> None:
        """加载模型配置文件"""
        models_file = Path(settings.models_config_path)
        if not models_file.exists():
            # 如果新路径不存在，尝试旧路径
            models_file = Path("models.json")
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
                # 如果没有指定默认模型，选择第一个
                self._default_model_id = list(self._models_config.keys())[0]

        except (json.JSONDecodeError, KeyError) as e:
            raise DefaultServerErrorException(f"模型配置文件格式错误: {str(e)}")

    def get_model_config(self, model_id: Optional[str] = None) -> ModelConfig:
        """获取模型配置"""
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        if model_id not in self._models_config:
            available_models = ", ".join(self._models_config.keys())
            raise InvalidParameterException(
                f"未知的模型: {model_id}，可用模型: {available_models}"
            )

        return self._models_config[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有可用模型"""
        models = []
        for model_id, config in self._models_config.items():
            # 检查模型文件是否存在
            model_path = Path(settings.MODELSCOPE_PATH) / config.path
            path_exists = model_path.exists()

            # 检查模型是否已加载
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
                    "path_exists": path_exists,
                }
            )

        return models

    def get_asr_engine(self, model_id: Optional[str] = None) -> ASREngine:
        """获取ASR引擎，支持缓存"""
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        # 如果已经加载，直接返回
        if model_id in self._loaded_engines:
            return self._loaded_engines[model_id]

        # 加载新模型
        config = self.get_model_config(model_id)
        engine = self._create_engine(config)

        # 缓存引擎
        self._loaded_engines[model_id] = engine

        return engine

    def _create_engine(self, config: ModelConfig) -> ASREngine:
        """根据配置创建ASR引擎"""
        if config.engine.lower() == "funasr":
            return FunASREngine(
                model_path=config.path,
                device=settings.DEVICE,
                vad_model=settings.VAD_MODEL,
                vad_model_revision=settings.VAD_MODEL_REVISION,
                punc_model=settings.PUNC_MODEL,
                punc_model_revision=settings.PUNC_MODEL_REVISION,
                spk_model=settings.SPK_MODEL,
            )
        elif config.engine.lower() == "dolphin":
            return DolphinEngine(
                model_path=config.path,
                size=config.size,
                device=settings.DEVICE,
            )
        else:
            raise InvalidParameterException(f"不支持的引擎类型: {config.engine}")

    def unload_model(self, model_id: str) -> bool:
        """卸载指定模型"""
        if model_id in self._loaded_engines:
            del self._loaded_engines[model_id]
            # 强制垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        memory_info = {
            "model_list": list(self._loaded_engines.keys()),
            "loaded_count": len(self._loaded_engines),
        }

        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
            }

        return memory_info

    def clear_cache(self) -> None:
        """清空模型缓存"""
        self._loaded_engines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 全局模型管理器实例
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
