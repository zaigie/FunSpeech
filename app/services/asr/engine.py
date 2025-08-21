# -*- coding: utf-8 -*-
"""
ASR引擎模块
封装ASR模型的加载和推理功能，支持多种ASR引擎
"""

import torch
import numpy as np
import logging
import threading
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from ...core.config import settings
from ...core.exceptions import ASRException, UnsupportedSampleRateException
from ...utils.audio import cleanup_temp_file
from ...utils.number_converter import apply_itn_to_text

logger = logging.getLogger(__name__)


class ASREngine(ABC):
    """ASR引擎抽象基类"""

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


class FunASREngine(ASREngine):
    """FunASR语音识别引擎"""

    def __init__(
        self,
        model_path: str = None,
        device: str = "auto",
        vad_model: str = None,
        vad_model_revision: str = "v2.0.4",
        punc_model: str = None,
        punc_model_revision: str = "v2.0.4",
        spk_model: str = None,
    ):
        from funasr import AutoModel

        self.model: Optional[AutoModel] = None
        self._device: str = self._detect_device(device)
        self.model_path = (
            model_path
            or "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        )
        self.vad_model = vad_model or settings.VAD_MODEL
        self.vad_model_revision = vad_model_revision
        self.punc_model = punc_model or settings.PUNC_MODEL
        self.punc_model_revision = punc_model_revision
        self.spk_model = spk_model or settings.SPK_MODEL
        self._load_model()

    def _detect_device(self, device: str = "auto") -> str:
        """检测可用设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                return "cpu"
        return device

    def _load_model(self) -> None:
        """加载FunASR模型"""
        try:
            from funasr import AutoModel

            # 构建模型参数
            model_kwargs = {
                "model": self.model_path,
                "trust_remote_code": True,
                "device": self._device,
            }

            # 添加VAD模型
            if self.vad_model and isinstance(self.vad_model, str):
                model_kwargs["vad_model"] = self.vad_model
                if self.vad_model_revision and isinstance(self.vad_model_revision, str):
                    model_kwargs["vad_model_revision"] = self.vad_model_revision

            # 添加标点模型
            if self.punc_model and isinstance(self.punc_model, str):
                model_kwargs["punc_model"] = self.punc_model
                if self.punc_model_revision and isinstance(
                    self.punc_model_revision, str
                ):
                    model_kwargs["punc_model_revision"] = self.punc_model_revision

            # 添加说话人分离模型（如果启用）
            if self.spk_model and isinstance(self.spk_model, str):
                model_kwargs["spk_model"] = self.spk_model

            self.model = AutoModel(**model_kwargs)

        except Exception as e:
            raise ASRException(50000001, f"FunASR模型加载失败: {str(e)}")

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
        """使用FunASR转录音频文件"""
        if not self.model:
            raise ASRException(50000001, "模型未加载")

        try:
            # FunASR的generate方法可以直接接受文件路径
            result = self.model.generate(
                input=audio_path,
                hotword=hotwords if hotwords else None,
                cache={} if not hasattr(self, "_cache") else getattr(self, "_cache"),
            )

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
            raise ASRException(50000002, f"语音识别失败: {str(e)}")

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device


class DolphinEngine(ASREngine):
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

    def _detect_device(self, device: str = "auto") -> str:
        """检测可用设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        # 转换设备格式，Dolphin使用 "cuda" 而不是 "cuda:0"
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

            # 使用dolphin库加载模型
            self.model = dolphin.load_model(self.size, model_path, self._device)

            logger.info("Dolphin模型加载成功")

        except Exception as e:
            raise ASRException(50000001, f"Dolphin模型加载失败: {str(e)}")

    def _clean_dolphin_text(self, text: str) -> str:
        """清理dolphin输出的特殊标记和时间戳"""
        import re

        if not text:
            return ""

        # 移除语言和区域标记，如: <zh><SHANGHAI><asr>
        text = re.sub(r"<[^>]*>", "", text)

        # 移除时间戳，如: <0.00> 和 <12.80>
        text = re.sub(r"<[\d.]+>", "", text)

        # 清理多余的空格
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _add_punctuation(self, text: str) -> str:
        """使用标点符号模型为文本添加标点"""
        if not text or not text.strip():
            return text

        try:
            # 使用全局标点符号模型缓存
            punc_model = get_global_punc_model(self._device)
            if punc_model is None:
                logger.warning("标点符号模型未加载，返回原文本")
                return text

            # 调用标点符号模型
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
            raise ASRException(50000001, "模型未加载")

        try:
            import dolphin

            if self.model is None:
                raise ASRException(50000001, "模型未加载")

            logger.info(f"开始Dolphin语音识别，文件: {audio_path}")
            logger.debug(
                f"识别参数: 语言={dolphin_lang_sym}, 区域={dolphin_region_sym}"
            )

            # 使用dolphin加载音频
            waveform = dolphin.load_audio(audio_path)

            # 执行语音识别
            if dolphin_lang_sym and dolphin_region_sym:
                result = self.model(
                    waveform,
                    lang_sym=dolphin_lang_sym,
                    region_sym=dolphin_region_sym,
                )
            else:
                result = self.model(waveform)

            # 提取识别文本
            recognition_text = result.text if hasattr(result, "text") else str(result)

            logger.debug(f"Dolphin原始识别结果: {recognition_text}")

            # 清理dolphin特殊标记
            cleaned_text = self._clean_dolphin_text(recognition_text)
            logger.debug(f"清理后文本: {cleaned_text}")

            # 如果启用标点符号，则添加标点
            if enable_punctuation and cleaned_text.strip():
                final_text = self._add_punctuation(cleaned_text)
            else:
                final_text = cleaned_text

            # 应用ITN处理
            if enable_itn and final_text:
                logger.debug(f"应用ITN处理前: {final_text}")
                final_text = apply_itn_to_text(final_text)
                logger.debug(f"应用ITN处理后: {final_text}")

            logger.info(f"Dolphin识别完成: {final_text}")

            return final_text

        except Exception as e:
            raise ASRException(50000002, f"语音识别失败: {str(e)}")

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device


# 全局ASR引擎实例缓存
_asr_engine: Optional[ASREngine] = None

# 全局标点符号模型缓存（避免重复加载）
_global_punc_model = None
_punc_model_lock = threading.Lock()


def get_global_punc_model(device: str):
    """获取全局标点符号模型实例，避免重复加载"""
    global _global_punc_model

    with _punc_model_lock:
        if _global_punc_model is None:
            try:
                from funasr import AutoModel

                logger.info("正在加载全局标点符号模型...")
                _global_punc_model = AutoModel(
                    model=settings.PUNC_MODEL,
                    model_revision=settings.PUNC_MODEL_REVISION,
                    device=device,
                )
                logger.info("全局标点符号模型加载成功")
            except Exception as e:
                logger.error(f"全局标点符号模型加载失败: {str(e)}")
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
            logger.info("全局标点符号模型缓存已清理")


def get_asr_engine() -> ASREngine:
    """获取全局ASR引擎实例（默认模型）"""
    global _asr_engine
    if _asr_engine is None:
        # 使用模型管理器获取默认引擎
        from .manager import get_model_manager

        model_manager = get_model_manager()
        _asr_engine = model_manager.get_asr_engine()
    return _asr_engine
