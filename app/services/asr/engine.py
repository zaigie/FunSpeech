# -*- coding: utf-8 -*-
"""
ASR引擎模块 - 支持多种ASR引擎
"""

import torch
import numpy as np
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple, Union
from abc import ABC, abstractmethod
from enum import Enum

from funasr import AutoModel

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from ...utils.audio import cleanup_temp_file
from ...utils.text_processing import apply_itn_to_text

logger = logging.getLogger(__name__)


def parse_gpu_config(gpu_str: str) -> Tuple[List[str], str]:
    """解析GPU配置字符串

    Args:
        gpu_str: GPU配置字符串
            - "" 或 "auto": 自动检测，返回单个设备
            - "cpu": 使用CPU
            - "0": 使用单卡 cuda:0
            - "0,1,2": 使用多卡

    Returns:
        (设备列表, 单设备字符串) - 多卡时设备列表有多个元素，单卡/CPU时只有一个
    """
    gpu_str = gpu_str.strip().lower() if gpu_str else ""

    # 空值或auto: 自动检测
    if not gpu_str or gpu_str == "auto":
        if torch.cuda.is_available():
            return ["cuda:0"], "cuda:0"
        else:
            return ["cpu"], "cpu"

    # CPU模式
    if gpu_str == "cpu":
        return ["cpu"], "cpu"

    # GPU列表模式
    devices = []
    for gpu_id in gpu_str.split(","):
        gpu_id = gpu_id.strip()
        if gpu_id and gpu_id.isdigit():
            devices.append(f"cuda:{gpu_id}")

    if not devices:
        # 无效配置，回退到自动检测
        if torch.cuda.is_available():
            return ["cuda:0"], "cuda:0"
        else:
            return ["cpu"], "cpu"

    return devices, devices[0]


class ModelType(Enum):
    """模型类型枚举"""

    OFFLINE = "offline"
    REALTIME = "realtime"


class BaseASREngine(ABC):
    """基础ASR引擎抽象基类"""

    @abstractmethod
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
        """转录音频文件"""
        pass

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """获取设备信息"""
        pass

    @property
    @abstractmethod
    def supports_realtime(self) -> bool:
        """是否支持实时识别"""
        pass

    def _detect_device(self, device: str = None) -> str:
        """检测可用设备"""
        if device and device != "auto":
            return device
        _, single_device = parse_gpu_config(settings.ASR_GPUS)
        return single_device


class RealTimeASREngine(BaseASREngine):
    """实时ASR引擎抽象基类"""

    @property
    def supports_realtime(self) -> bool:
        """支持实时识别"""
        return True

    @abstractmethod
    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        """WebSocket流式语音识别"""
        pass


class FunASREngine(RealTimeASREngine):
    """FunASR语音识别引擎"""

    def __init__(
        self,
        offline_model_path: Optional[str] = None,
        realtime_model_path: Optional[str] = None,
        device: str = "auto",
        vad_model: str = None,
        vad_model_revision: str = "v2.0.4",
        punc_model: str = None,
        punc_model_revision: str = "v2.0.4",
        punc_realtime_model: str = None,
    ):
        self.offline_model: Optional[AutoModel] = None
        self.realtime_model: Optional[AutoModel] = None
        self.punc_model_instance: Optional[AutoModel] = None
        self.punc_realtime_model_instance: Optional[AutoModel] = None
        self._device: str = self._detect_device(device)

        # 模型路径配置
        self.offline_model_path = offline_model_path
        self.realtime_model_path = realtime_model_path

        # 辅助模型配置
        self.vad_model = vad_model or settings.VAD_MODEL
        self.vad_model_revision = vad_model_revision
        self.punc_model = punc_model or settings.PUNC_MODEL
        self.punc_model_revision = punc_model_revision
        self.punc_realtime_model = punc_realtime_model or settings.PUNC_REALTIME_MODEL

        self._load_models_based_on_mode()

    def _load_models_based_on_mode(self) -> None:
        """根据ASR_MODEL_MODE加载对应的模型"""
        mode = settings.ASR_MODEL_MODE.lower()

        if mode == "all":
            # 加载所有可用模型
            if self.offline_model_path:
                self._load_offline_model()
            if self.realtime_model_path:
                self._load_realtime_model()
        elif mode == "offline":
            # 只加载离线模型
            if self.offline_model_path:
                self._load_offline_model()
            else:
                logger.warning("ASR_MODEL_MODE设置为offline，但未提供离线模型路径")
        elif mode == "realtime":
            # 只加载实时模型
            if self.realtime_model_path:
                self._load_realtime_model()
            else:
                logger.warning("ASR_MODEL_MODE设置为realtime，但未提供实时模型路径")
        else:
            raise DefaultServerErrorException(f"不支持的ASR_MODEL_MODE: {mode}")

    def _load_offline_model(self) -> None:
        """加载离线FunASR模型（不再内嵌VAD/PUNC，改用全局实例）"""
        try:
            logger.info(f"正在加载离线FunASR模型: {self.offline_model_path}")

            model_kwargs = {
                "model": self.offline_model_path,
                "device": self._device,
                **settings.FUNASR_AUTOMODEL_KWARGS,
            }

            self.offline_model = AutoModel(**model_kwargs)
            logger.info("离线FunASR模型加载成功（VAD/PUNC将按需使用全局实例）")

        except Exception as e:
            raise DefaultServerErrorException(f"离线FunASR模型加载失败: {str(e)}")

    def _load_realtime_model(self) -> None:
        """加载实时FunASR模型（不再内嵌PUNC，改用全局实例）"""
        try:
            logger.info(f"正在加载实时FunASR模型: {self.realtime_model_path}")

            model_kwargs = {
                "model": self.realtime_model_path,
                "device": self._device,
                **settings.FUNASR_AUTOMODEL_KWARGS,
            }

            self.realtime_model = AutoModel(**model_kwargs)
            logger.info("实时FunASR模型加载成功（PUNC将按需使用全局实例）")

        except Exception as e:
            raise DefaultServerErrorException(f"实时FunASR模型加载失败: {str(e)}")

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
        """使用FunASR转录音频文件（支持动态启用VAD和PUNC）

        根据参数组合采用不同策略：
        1. 只PUNC：手动后处理
        2. 有VAD：利用全局实例直接构造临时AutoModel（复用已加载模型）
        """
        # 优先使用离线模型进行文件识别
        if not self.offline_model:
            raise DefaultServerErrorException(
                "离线模型未加载，无法进行文件识别。"
                "请将 ASR_MODEL_MODE 设置为 offline 或 all"
            )

        try:
            # 根据参数决定是否需要VAD/PUNC
            need_vad = enable_vad
            need_punc = enable_punctuation

            if need_vad:
                # 使用VAD时，需要构建临时AutoModel
                # 预加载全局VAD和PUNC实例
                logger.debug("启用VAD，预加载全局VAD模型")
                vad_model_instance = get_global_vad_model(self._device)

                punc_model_instance = None
                if need_punc:
                    logger.debug("预加载全局PUNC模型")
                    punc_model_instance = get_global_punc_model(self._device)

                # 创建临时AutoModel（直接赋值已加载的模型，而不是重新构建）
                temp_automodel = type("TempAutoModel", (), {})()
                temp_automodel.model = self.offline_model.model
                temp_automodel.kwargs = self.offline_model.kwargs
                temp_automodel.model_path = self.offline_model.model_path

                # HACK: 不设置的话，会导致临时AutoModel的spk_model属性不存在而报错
                temp_automodel.spk_model = None

                # 设置VAD（使用全局实例）
                temp_automodel.vad_model = vad_model_instance.model
                temp_automodel.vad_kwargs = vad_model_instance.kwargs

                # 设置PUNC（使用全局实例）
                if punc_model_instance:
                    temp_automodel.punc_model = punc_model_instance.model
                    temp_automodel.punc_kwargs = punc_model_instance.kwargs
                else:
                    temp_automodel.punc_model = None
                    temp_automodel.punc_kwargs = {}

                # 绑定方法（使用types.MethodType更可靠）
                import types

                temp_automodel.inference = types.MethodType(
                    AutoModel.inference, temp_automodel
                )
                temp_automodel.inference_with_vad = types.MethodType(
                    AutoModel.inference_with_vad, temp_automodel
                )
                temp_automodel.generate = types.MethodType(
                    AutoModel.generate, temp_automodel
                )

                logger.debug("临时AutoModel构建完成，调用generate")
                result = temp_automodel.generate(
                    input=audio_path,
                    hotword=hotwords if hotwords else None,
                    cache={},
                )
            else:
                # 不使用VAD，直接识别
                result = self.offline_model.generate(
                    input=audio_path,
                    hotword=hotwords if hotwords else None,
                    cache={},
                )

                # 如果启用了PUNC但没有VAD，需要手动应用PUNC
                if need_punc and result and len(result) > 0:
                    text = result[0].get("text", "").strip()
                    if text:
                        logger.debug("手动应用PUNC模型（因为未启用VAD）")
                        punc_model_instance = get_global_punc_model(self._device)
                        punc_result = punc_model_instance.generate(
                            input=text,
                            cache={},
                        )
                        if punc_result and len(punc_result) > 0:
                            result[0]["text"] = punc_result[0].get("text", text)
                            logger.debug(f"标点符号添加完成")

            # 提取识别结果
            if result and len(result) > 0:
                text = result[0].get("text", "")
                text = text.strip()

                # 应用ITN处理
                if enable_itn and text:
                    logger.debug(f"应用ITN处理前: {text}")
                    text = apply_itn_to_text(text)
                    logger.debug(f"应用ITN处理后: {text}")

                return text
            else:
                return ""

        except Exception as e:
            raise DefaultServerErrorException(f"语音识别失败: {str(e)}")

    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        """WebSocket流式语音识别（未实现）"""
        if not self.realtime_model:
            raise DefaultServerErrorException(
                "实时模型未加载，无法进行WebSocket流式识别。"
                "请将 ASR_MODEL_MODE 设置为 realtime 或 all"
            )

        logger.warning("WebSocket流式识别功能尚未实现")
        return ""

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.offline_model is not None or self.realtime_model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device


class DolphinEngine(BaseASREngine):
    """Dolphin语音识别引擎"""

    def __init__(
        self,
        model_path: str = None,
        size: str = "small",
        device: str = "auto",
    ):
        self.model = None
        self._device: str = self._detect_device(device)
        self.model_path = model_path or "DataoceanAI/dolphin-small"
        self.size = size
        self._load_model()

    @property
    def supports_realtime(self) -> bool:
        """Dolphin不支持实时识别"""
        return False

    def _detect_device(self, device: str = "auto") -> str:
        """检测可用设备"""
        if device == "auto":
            _, single_device = parse_gpu_config(settings.ASR_GPUS)
            # Dolphin 使用 "cuda" 而不是 "cuda:0"
            if single_device.startswith("cuda:"):
                return "cuda"
            return single_device
        if device.startswith("cuda:"):
            return "cuda"
        return device

    def _load_model(self) -> None:
        """加载Dolphin模型"""
        try:
            import dolphin
            import os

            logger.info(f"正在加载Dolphin模型，设备: {self._device}")
            logger.info(f"模型大小: {self.size}")
            model_path = os.path.join(settings.MODELSCOPE_PATH, self.model_path)
            logger.info(f"模型路径: {model_path}")

            self.model = dolphin.load_model(self.size, model_path, self._device)

            logger.info("Dolphin模型加载成功")

        except Exception as e:
            raise DefaultServerErrorException(f"Dolphin模型加载失败: {str(e)}")

    def _clean_dolphin_text(self, text: str) -> str:
        """清理dolphin输出的特殊标记"""
        import re

        if not text:
            return ""

        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"<[\d.]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _add_punctuation(self, text: str) -> str:
        """使用标点符号模型为文本添加标点"""
        if not text or not text.strip():
            return text

        try:
            punc_model = get_global_punc_model(self._device)
            if punc_model is None:
                logger.warning("标点符号模型未加载，返回原文本")
                return text

            result = punc_model.generate(text)

            if result and len(result) > 0:
                punctuated_text = result[0].get("text", text)
                logger.debug(f"标点符号处理: {text} -> {punctuated_text}")
                return punctuated_text
            else:
                return text

        except Exception as e:
            logger.warning(f"标点符号处理失败: {str(e)}")
            return text

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
        """使用Dolphin转录音频文件"""
        if not self.model:
            raise DefaultServerErrorException("模型未加载")

        try:
            import dolphin

            if self.model is None:
                raise DefaultServerErrorException("模型未加载")

            logger.debug(f"开始Dolphin语音识别，文件: {audio_path}")
            logger.debug(
                f"识别参数: 语言={dolphin_lang_sym}, 区域={dolphin_region_sym}"
            )

            waveform = dolphin.load_audio(audio_path)

            if dolphin_lang_sym and dolphin_region_sym:
                result = self.model(
                    waveform,
                    lang_sym=dolphin_lang_sym,
                    region_sym=dolphin_region_sym,
                )
            else:
                result = self.model(waveform)

            recognition_text = result.text if hasattr(result, "text") else str(result)
            logger.debug(f"Dolphin原始识别结果: {recognition_text}")

            cleaned_text = self._clean_dolphin_text(recognition_text)
            logger.debug(f"清理后文本: {cleaned_text}")

            if enable_punctuation and cleaned_text.strip():
                final_text = self._add_punctuation(cleaned_text)
            else:
                final_text = cleaned_text

            if enable_itn and final_text:
                logger.debug(f"应用ITN处理前: {final_text}")
                final_text = apply_itn_to_text(final_text)
                logger.debug(f"应用ITN处理后: {final_text}")

            logger.debug(f"Dolphin识别完成: {final_text}")

            return final_text

        except Exception as e:
            raise DefaultServerErrorException(f"语音识别失败: {str(e)}")

    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        """WebSocket流式语音识别（Dolphin不支持）"""
        raise DefaultServerErrorException("Dolphin引擎不支持WebSocket流式识别")

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device


# 全局ASR引擎实例缓存
_asr_engine: Optional[BaseASREngine] = None

# 全局VAD模型缓存（避免重复加载）
_global_vad_model = None
_vad_model_lock = threading.Lock()

# 全局标点符号模型缓存（避免重复加载）
_global_punc_model = None
_punc_model_lock = threading.Lock()

# 全局实时标点符号模型缓存（避免重复加载）
_global_punc_realtime_model = None
_punc_realtime_model_lock = threading.Lock()


def get_global_vad_model(device: str):
    """获取全局VAD模型实例"""
    global _global_vad_model

    with _vad_model_lock:
        if _global_vad_model is None:
            try:
                logger.info("正在加载全局VAD模型...")

                _global_vad_model = AutoModel(
                    model=settings.VAD_MODEL,
                    model_revision=settings.VAD_MODEL_REVISION,
                    device=device,
                    **settings.FUNASR_AUTOMODEL_KWARGS,
                )
                logger.info("全局VAD模型加载成功")
            except Exception as e:
                logger.error(f"全局VAD模型加载失败: {str(e)}")
                _global_vad_model = None
                raise

    return _global_vad_model


def clear_global_vad_model():
    """清理全局VAD模型缓存"""
    global _global_vad_model

    with _vad_model_lock:
        if _global_vad_model is not None:
            del _global_vad_model
            _global_vad_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局VAD模型缓存已清理")


def get_global_punc_model(device: str):
    """获取全局标点符号模型实例（离线版）"""
    global _global_punc_model

    with _punc_model_lock:
        if _global_punc_model is None:
            try:
                logger.info("正在加载全局标点符号模型（离线）...")

                _global_punc_model = AutoModel(
                    model=settings.PUNC_MODEL,
                    model_revision=settings.PUNC_MODEL_REVISION,
                    device=device,
                    **settings.FUNASR_AUTOMODEL_KWARGS,
                )
                logger.info("全局标点符号模型（离线）加载成功")
            except Exception as e:
                logger.error(f"全局标点符号模型（离线）加载失败: {str(e)}")
                _global_punc_model = None
                raise

    return _global_punc_model


def clear_global_punc_model():
    """清理全局标点符号模型缓存"""
    global _global_punc_model

    with _punc_model_lock:
        if _global_punc_model is not None:
            del _global_punc_model
            _global_punc_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局标点符号模型（离线）缓存已清理")


def get_global_punc_realtime_model(device: str):
    """获取全局实时标点符号模型实例"""
    global _global_punc_realtime_model

    with _punc_realtime_model_lock:
        if _global_punc_realtime_model is None:
            try:
                logger.info("正在加载全局标点符号模型（实时）...")

                _global_punc_realtime_model = AutoModel(
                    model=settings.PUNC_REALTIME_MODEL,
                    model_revision=settings.PUNC_MODEL_REVISION,
                    device=device,
                    **settings.FUNASR_AUTOMODEL_KWARGS,
                )
                logger.info("全局标点符号模型（实时）加载成功")
            except Exception as e:
                logger.error(f"全局标点符号模型（实时）加载失败: {str(e)}")
                _global_punc_realtime_model = None
                raise

    return _global_punc_realtime_model


def clear_global_punc_realtime_model():
    """清理全局实时标点符号模型缓存"""
    global _global_punc_realtime_model

    with _punc_realtime_model_lock:
        if _global_punc_realtime_model is not None:
            del _global_punc_realtime_model
            _global_punc_realtime_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局标点符号模型（实时）缓存已清理")


def get_asr_engine() -> BaseASREngine:
    """获取全局ASR引擎实例"""
    global _asr_engine
    if _asr_engine is None:
        from .manager import get_model_manager

        model_manager = get_model_manager()
        _asr_engine = model_manager.get_asr_engine()
    return _asr_engine


class MultiGPUASREngine(RealTimeASREngine):
    """多GPU多副本ASR引擎，支持负载均衡"""

    def __init__(
        self,
        engine_factory,
        engine_kwargs: Dict[str, Any],
    ):
        """
        初始化多GPU ASR引擎

        Args:
            engine_factory: 引擎工厂函数，用于创建单个引擎实例
            engine_kwargs: 创建引擎时的基础参数（不含device）
        """
        self._engine_factory = engine_factory
        self._engine_kwargs = engine_kwargs
        self._engines: List[BaseASREngine] = []
        self._devices: List[str] = []
        self._lock = threading.Lock()
        self._current_index = 0
        self._engine_active_count: List[int] = []

        self._init_engines()

    def _init_engines(self):
        """初始化所有GPU上的引擎实例"""
        # 解析GPU配置
        gpu_devices, single_device = parse_gpu_config(settings.ASR_GPUS)

        # 单设备模式（包括CPU、单卡GPU、自动检测）
        if len(gpu_devices) == 1:
            logger.info(f"ASR使用单设备模式: {single_device}")
            kwargs = self._engine_kwargs.copy()
            kwargs["device"] = single_device
            engine = self._engine_factory(**kwargs)
            self._engines.append(engine)
            self._devices.append(engine.device)
            self._engine_active_count.append(0)
            return

        # 多GPU模式
        if not torch.cuda.is_available():
            raise DefaultServerErrorException("配置了多GPU但CUDA不可用")

        available_gpus = torch.cuda.device_count()
        logger.info(f"系统可用GPU数量: {available_gpus}, 配置的GPU设备: {gpu_devices}")

        # 在每个GPU上创建引擎实例
        for device in gpu_devices:
            gpu_id = int(device.split(":")[1])
            if gpu_id >= available_gpus:
                logger.warning(f"GPU {gpu_id} 不存在，跳过")
                continue

            logger.info(f"正在初始化 {device} 上的ASR引擎...")
            try:
                kwargs = self._engine_kwargs.copy()
                kwargs["device"] = device
                engine = self._engine_factory(**kwargs)
                self._engines.append(engine)
                self._devices.append(device)
                self._engine_active_count.append(0)
                logger.info(f"{device} 上的ASR引擎初始化成功")
            except Exception as e:
                logger.error(f"{device} 上的ASR引擎初始化失败: {e}")

        if not self._engines:
            raise DefaultServerErrorException("没有成功初始化任何ASR引擎")

        logger.info(f"多GPU ASR引擎初始化完成，共 {len(self._engines)} 个副本")

    def _select_engine(self) -> Tuple[int, BaseASREngine]:
        """选择一个引擎（最少连接数策略）

        Returns:
            (引擎索引, 引擎实例)
        """
        with self._lock:
            # 选择当前活跃请求数最少的引擎
            min_count = min(self._engine_active_count)
            for i, count in enumerate(self._engine_active_count):
                if count == min_count:
                    self._engine_active_count[i] += 1
                    return i, self._engines[i]

            # 降级到轮询
            idx = self._current_index
            self._current_index = (self._current_index + 1) % len(self._engines)
            self._engine_active_count[idx] += 1
            return idx, self._engines[idx]

    def _release_engine(self, index: int):
        """释放引擎（减少活跃计数）"""
        with self._lock:
            if 0 <= index < len(self._engine_active_count):
                self._engine_active_count[index] = max(0, self._engine_active_count[index] - 1)

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
        """转录音频文件（负载均衡）"""
        idx, engine = self._select_engine()
        try:
            logger.debug(f"使用引擎 {idx} ({self._devices[idx]}) 处理ASR请求")
            return engine.transcribe_file(
                audio_path=audio_path,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                enable_itn=enable_itn,
                enable_vad=enable_vad,
                sample_rate=sample_rate,
                dolphin_lang_sym=dolphin_lang_sym,
                dolphin_region_sym=dolphin_region_sym,
            )
        finally:
            self._release_engine(idx)

    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs,
    ) -> str:
        """WebSocket流式语音识别（负载均衡）

        注意：WebSocket流式识别需要保持会话状态，因此需要在更上层
        （如WebSocket连接建立时）选择引擎并在整个会话期间保持使用同一个引擎。
        这里的实现假设调用者会正确处理会话管理。
        """
        # 对于流式识别，由于需要保持会话状态，这里提供一个简单实现
        # 实际使用时应该在会话级别管理引擎选择
        idx, engine = self._select_engine()
        try:
            if isinstance(engine, RealTimeASREngine):
                return engine.transcribe_websocket(
                    audio_chunk=audio_chunk,
                    cache=cache,
                    is_final=is_final,
                    **kwargs,
                )
            else:
                raise DefaultServerErrorException("选中的引擎不支持实时识别")
        finally:
            self._release_engine(idx)

    def get_engine_for_session(self) -> Tuple[int, BaseASREngine]:
        """为WebSocket会话获取一个引擎（会话级别的负载均衡）

        返回:
            (引擎索引, 引擎实例) - 调用者负责在会话结束时调用 release_session_engine
        """
        return self._select_engine()

    def release_session_engine(self, index: int):
        """释放WebSocket会话使用的引擎"""
        self._release_engine(index)

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return any(engine.is_model_loaded() for engine in self._engines)

    @property
    def device(self) -> str:
        """获取设备信息（返回所有设备）"""
        return ",".join(self._devices)

    @property
    def supports_realtime(self) -> bool:
        """是否支持实时识别"""
        return any(
            isinstance(engine, RealTimeASREngine) and engine.supports_realtime
            for engine in self._engines
        )

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎状态统计"""
        with self._lock:
            return {
                "total_engines": len(self._engines),
                "devices": self._devices,
                "active_counts": self._engine_active_count.copy(),
            }
