# -*- coding: utf-8 -*-
"""
TTS引擎模块
"""

import sys
import logging
from typing import List, Dict, Any
from pathlib import Path

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from ...utils.audio import save_audio_array, generate_temp_audio_path

logger = logging.getLogger(__name__)


class CosyVoiceTTSEngine:
    """CosyVoice TTS引擎"""

    def __init__(self, load_sft: bool = True, load_clone: bool = True):
        """
        初始化TTS引擎

        Args:
            load_sft: 是否加载SFT模型，默认为True
            load_clone: 是否加载零样本克隆模型，默认为True
        """
        # 两个不同的模型实例
        self.cosyvoice_sft = None  # 用于预设音色（SFT模式=CosyVoice）
        self.cosyvoice_clone = None  # 用于音零样本色克隆（零样本/跨语言=CosyVoice2）

        self._device = self._detect_device()
        self._sft_model_loaded = False
        self._clone_model_loaded = False
        self._preset_voices = settings.PRESET_VOICES.copy()
        self._voice_manager = None  # 新的音色管理器
        self._load_sft = load_sft  # 保存配置
        self._load_clone = load_clone  # 保存配置
        self._setup_paths()
        self._load_models()

    def _detect_device(self) -> str:
        """检测可用设备"""
        import torch

        device = settings.TTS_DEVICE
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            else:
                return "cpu"
        return device

    def _setup_paths(self):
        """设置第三方库路径"""
        # 添加CosyVoice路径到Python路径
        cosyvoice_path = Path(__file__).parent / "third_party" / "CosyVoice"
        if cosyvoice_path.exists():
            sys.path.insert(0, str(cosyvoice_path))

        # 添加Matcha-TTS路径到Python路径
        matcha_path = cosyvoice_path / "third_party" / "Matcha-TTS"
        if matcha_path.exists():
            sys.path.insert(0, str(matcha_path))

    def _load_models(self):
        """加载TTS模型"""
        logger.info("开始加载TTS模型...")
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

            fp16_enabled = self._device.startswith("cuda")

            # 根据配置决定是否加载SFT模型（用于预设音色）
            if self._load_sft:
                try:
                    logger.info("正在加载SFT模型（CosyVoice1）...")
                    self.cosyvoice_sft = CosyVoice(
                        settings.SFT_MODEL_ID,
                        load_jit=True,
                        load_trt=True,
                        fp16=fp16_enabled,
                    )
                    self._sft_model_loaded = True
                    logger.info("SFT模型加载成功")
                except Exception as e:
                    logger.warning(f"SFT模型加载失败: {str(e)}")
                    self._sft_model_loaded = False
            else:
                logger.info("跳过SFT模型加载（load_sft=False）")
                self._sft_model_loaded = False

            # 加载零样本克隆模型（用于零样本和跨语言）
            if self._load_clone:
                try:
                    logger.info("正在加载零样本克隆模型（CosyVoice2）...")
                    self.cosyvoice_clone = CosyVoice2(
                        settings.CLONE_MODEL_ID,
                        load_jit=True,
                        load_trt=True,
                        load_vllm=False,
                        fp16=fp16_enabled,
                    )
                    self._clone_model_loaded = True
                    logger.info("零样本克隆模型加载成功")
                except Exception as e:
                    logger.warning(f"零样本克隆模型加载失败: {str(e)}")
                    self._clone_model_loaded = False
            else:
                logger.info("跳过零样本克隆模型加载（load_clone=False）")
                self._clone_model_loaded = False

            if not self._sft_model_loaded and not self._clone_model_loaded:
                raise DefaultServerErrorException("所有模型都加载失败")

            # 初始化音色管理器（只有零样本克隆模型加载成功时才初始化）
            if self._clone_model_loaded:
                logger.info("正在初始化音色管理器...")
                self._load_voice_manager()

            logger.info("TTS模型加载完成")

        except Exception as e:
            logger.error(f"TTS模型加载失败: {str(e)}")
            raise DefaultServerErrorException(f"TTS模型加载失败: {str(e)}")

    def _load_voice_manager(self):
        """初始化音色管理器"""
        try:
            from .clone import VoiceManager

            self._voice_manager = VoiceManager(self.cosyvoice_clone)

            # 获取零样本克隆音色列表并添加到预设音色中（一次性加载）
            clone_voices = self._voice_manager.list_clone_voices()
            for voice in clone_voices:
                if voice not in self._preset_voices:
                    self._preset_voices.append(voice)

            logger.info(
                f"音色管理器已初始化，发现 {len(clone_voices)} 个零样本克隆音色"
            )
        except Exception as e:
            logger.warning(f"初始化音色管理器失败: {str(e)}")

    def synthesize_speech(
        self,
        text: str,
        voice: str = "中文女",
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int = 22050,
        volume: int = 50,
        prompt: str = "",
    ) -> str:
        """语音合成（自动判断音色类型）"""
        try:
            # 检查是否为零样本克隆音色
            if self._voice_manager and self._voice_manager.is_voice_available(voice):
                # 检查是否在零样本克隆音色列表中
                if voice in self._voice_manager.list_clone_voices():
                    sample_rate = 24000
                    logger.info(f"使用零样本克隆音色模型合成: {voice}")
                    return self._synthesize_with_saved_voice(
                        text, voice, speed, format, sample_rate, volume, prompt
                    )

            # 使用预训练音色合成
            return self.synthesize_with_preset_voice(
                text, voice, speed, format, sample_rate, volume
            )

        except Exception as e:
            raise DefaultServerErrorException(f"语音合成失败: {str(e)}")

    def synthesize_with_preset_voice(
        self,
        text: str,
        voice: str = "中文女",
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int = 22050,
        volume: int = 50,
    ) -> str:
        """使用预设音色合成语音"""
        # 检查SFT模型是否可用
        if not self.cosyvoice_sft:
            model_mode = settings.TTS_MODEL_MODE.lower()
            if model_mode == "cosyvoice2":
                raise DefaultServerErrorException(
                    "当前配置为仅使用零样本克隆模型（TTS_MODEL_MODE=cosyvoice2），无法使用预设音色。"
                    "如需使用预设音色，请设置环境变量 TTS_MODEL_MODE=all 或 TTS_MODEL_MODE=cosyvoice1"
                )
            else:
                raise DefaultServerErrorException("SFT模型未加载")

        # 使用CosyVoice SFT模型进行预设音色合成
        logger.info(
            f"使用预训练音色模型合成: {voice}, 格式: {format}, 采样率: {sample_rate}"
        )

        # 收集所有音频片段
        audio_segments = []
        for audio_data in self.cosyvoice_sft.inference_sft(
            text, voice, stream=False, speed=speed
        ):
            audio_segments.append(audio_data["tts_speech"].numpy())

        if not audio_segments:
            raise DefaultServerErrorException("音频合成失败，未生成任何音频片段")

        # 如果有多个音频片段，需要拼接它们
        if len(audio_segments) > 1:
            import numpy as np

            # 拼接所有音频片段
            combined_audio = np.concatenate(audio_segments, axis=1)
            logger.info(f"合并了 {len(audio_segments)} 个音频片段")
        else:
            combined_audio = audio_segments[0]

        # 保存音频文件，使用指定的格式和采样率
        output_path = generate_temp_audio_path("preset_voice", f".{format}")
        save_audio_array(
            combined_audio,
            output_path,
            sample_rate=sample_rate,
            format=format,
            original_sr=self.cosyvoice_sft.sample_rate,
            volume=volume,
        )
        return output_path

    def _synthesize_with_saved_voice(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int = 22050,
        volume: int = 50,
        prompt: str = "",
    ) -> str:
        """使用保存的音色合成语音（基于官方API）"""
        if not self.cosyvoice_clone:
            model_mode = settings.TTS_MODEL_MODE.lower()
            if model_mode == "cosyvoice1":
                raise DefaultServerErrorException(
                    "当前配置为仅使用SFT模型（TTS_MODEL_MODE=cosyvoice1），无法使用零样本音色克隆功能。"
                    "如需使用零样本音色克隆，请设置环境变量 TTS_MODEL_MODE=all 或 TTS_MODEL_MODE=cosyvoice2"
                )
            else:
                raise DefaultServerErrorException("零样本克隆模型未加载")

        try:
            # 收集所有音频片段
            audio_segments = []
            # 使用官方API进行音色合成 - 直接通过zero_shot_spk_id引用保存的音色
            for audio_data in self.cosyvoice_clone.inference_zero_shot(
                text,
                prompt,  # 使用传入的prompt_text
                None,  # 不需要音频
                zero_shot_spk_id=voice,  # 使用保存的音色ID
                stream=False,
                speed=speed,
            ):
                audio_segments.append(audio_data["tts_speech"].numpy())

            if not audio_segments:
                raise DefaultServerErrorException("音频合成失败，未生成任何音频片段")

            # 如果有多个音频片段，需要拼接它们
            if len(audio_segments) > 1:
                import numpy as np

                # 拼接所有音频片段
                combined_audio = np.concatenate(audio_segments, axis=1)
                logger.info(f"合并了 {len(audio_segments)} 个音频片段")
            else:
                combined_audio = audio_segments[0]

            # 保存音频文件，使用指定的格式和采样率
            output_path = generate_temp_audio_path("saved_voice", f".{format}")
            save_audio_array(
                combined_audio,
                output_path,
                sample_rate=sample_rate,
                format=format,
                original_sr=self.cosyvoice_clone.sample_rate,
                volume=volume,
            )

            return output_path

        except Exception as e:
            raise DefaultServerErrorException(f"保存音色合成失败: {str(e)}")

    def get_voices(self) -> List[str]:
        """获取音色列表（根据模型加载模式返回对应音色）"""
        model_mode = settings.TTS_MODEL_MODE.lower()

        if model_mode == "cosyvoice1":
            # 仅返回预设音色
            return settings.PRESET_VOICES.copy()
        elif model_mode == "cosyvoice2":
            # 仅返回零样本克隆音色
            if self._voice_manager:
                return self._voice_manager.list_clone_voices()
            else:
                return []  # 允许空列表
        else:
            # all模式或其他：返回所有音色
            return self._preset_voices.copy()

    def get_voices_info(self) -> Dict[str, Dict[str, Any]]:
        """获取音色详细信息（根据模型加载模式返回对应音色信息）"""
        voices_info = {}
        model_mode = settings.TTS_MODEL_MODE.lower()

        # 预设音色信息
        preset_info = {
            "中文女": {
                "type": "preset",
                "language": "zh-CN",
                "gender": "female",
                "description": "标准中文女声",
            },
            "中文男": {
                "type": "preset",
                "language": "zh-CN",
                "gender": "male",
                "description": "标准中文男声",
            },
            "英文女": {
                "type": "preset",
                "language": "en-US",
                "gender": "female",
                "description": "标准英文女声",
            },
            "英文男": {
                "type": "preset",
                "language": "en-US",
                "gender": "male",
                "description": "标准英文男声",
            },
            "日语男": {
                "type": "preset",
                "language": "ja-JP",
                "gender": "male",
                "description": "标准日语男声",
            },
            "韩语女": {
                "type": "preset",
                "language": "ko-KR",
                "gender": "female",
                "description": "标准韩语女声",
            },
            "粤语女": {
                "type": "preset",
                "language": "zh-HK",
                "gender": "female",
                "description": "标准粤语女声",
            },
        }

        # 根据模型模式决定返回哪些音色信息
        if model_mode == "cosyvoice1":
            # 仅返回预设音色信息
            target_voices = settings.PRESET_VOICES
        elif model_mode == "cosyvoice2":
            # 仅返回零样本克隆音色信息
            target_voices = (
                self._voice_manager.list_clone_voices() if self._voice_manager else []
            )
        else:
            # all模式：返回所有音色信息
            target_voices = self._preset_voices

        for voice in target_voices:
            if voice in preset_info:
                info = preset_info[voice].copy()
            elif (
                self._voice_manager and voice in self._voice_manager.list_clone_voices()
            ):
                voice_info = self._voice_manager.get_voice_info(voice)
                info = {
                    "type": "clone",
                    "language": "zh-CN",
                    "gender": "unknown",
                    "description": f"零样本克隆音色：{voice}",
                    "reference_text": (
                        voice_info.get("reference_text", "") if voice_info else ""
                    ),
                    "audio_file": (
                        voice_info.get("audio_file", "") if voice_info else ""
                    ),
                    "added_at": voice_info.get("added_at", "") if voice_info else "",
                }
            else:
                info = {
                    "type": "custom",
                    "language": "unknown",
                    "gender": "unknown",
                    "description": f"自定义音色：{voice}",
                }

            info.update({"name": voice, "sample_rate": 22050, "available": True})
            voices_info[voice] = info

        return voices_info

    def refresh_voices(self):
        """刷新音色配置（根据模型模式仅在必要时重新加载）"""
        model_mode = settings.TTS_MODEL_MODE.lower()

        if model_mode == "cosyvoice1":
            # 仅使用预设音色
            self._preset_voices = settings.PRESET_VOICES.copy()
            logger.info(
                f"音色配置已刷新（cosyvoice1模式），当前有 {len(self._preset_voices)} 个预设音色"
            )
        elif model_mode == "cosyvoice2":
            # 仅使用零样本克隆音色
            if self._voice_manager:
                clone_voices = self._voice_manager.list_clone_voices()
                self._preset_voices = clone_voices.copy()
                logger.info(
                    f"音色配置已刷新（cosyvoice2模式），当前有 {len(clone_voices)} 个零样本克隆音色"
                )
            else:
                self._preset_voices = []
                logger.info("音色配置已刷新（cosyvoice2模式），当前没有可用音色")
        else:
            # all模式：包含所有音色
            self._preset_voices = settings.PRESET_VOICES.copy()
            if self._voice_manager:
                # 只从注册表获取零样本克隆音色，避免触发模型重新加载
                clone_voices = self._voice_manager.list_clone_voices()
                for voice in clone_voices:
                    if voice not in self._preset_voices:
                        self._preset_voices.append(voice)
            logger.info(
                f"音色配置已刷新（all模式），当前有 {len(self._preset_voices)} 个音色"
            )

    def is_sft_model_loaded(self) -> bool:
        """检查SFT模型是否已加载"""
        return self._sft_model_loaded

    def is_clone_model_loaded(self) -> bool:
        """检查零样本克隆模型是否已加载"""
        return self._clone_model_loaded

    def is_tts_model_loaded(self) -> bool:
        """检查TTS模型是否已加载"""
        return self._sft_model_loaded or self._clone_model_loaded

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device

    @property
    def voice_manager(self):
        """获取音色管理器"""
        return self._voice_manager


# 全局TTS引擎实例字典，根据配置保存不同的实例
_tts_engines: Dict[str, CosyVoiceTTSEngine] = {}


def get_tts_engine() -> CosyVoiceTTSEngine:
    """
    获取TTS引擎实例（根据环境变量TTS_MODEL_MODE决定加载策略）

    Returns:
        CosyVoiceTTSEngine: TTS引擎实例
    """
    global _tts_engines

    # 根据环境变量决定加载策略
    model_mode = settings.TTS_MODEL_MODE.lower()

    # 使用模型模式作为缓存key
    cache_key = f"engine_{model_mode}"

    if cache_key not in _tts_engines:
        if model_mode == "cosyvoice1":
            logger.info("创建TTS引擎实例，仅加载SFT模型（CosyVoice1）")
            _tts_engines[cache_key] = CosyVoiceTTSEngine(
                load_sft=True, load_clone=False
            )
        elif model_mode == "cosyvoice2":
            logger.info("创建TTS引擎实例，仅加载零样本克隆模型（CosyVoice2）")
            _tts_engines[cache_key] = CosyVoiceTTSEngine(
                load_sft=False, load_clone=True
            )
        elif model_mode == "all":
            logger.info("创建TTS引擎实例，加载所有模型")
            _tts_engines[cache_key] = CosyVoiceTTSEngine(load_sft=True, load_clone=True)
        else:
            logger.warning(f"未知的TTS_MODEL_MODE: {model_mode}，使用默认配置（all）")
            _tts_engines[cache_key] = CosyVoiceTTSEngine(load_sft=True, load_clone=True)

    return _tts_engines[cache_key]


def reset_tts_engines():
    """重置所有TTS引擎实例（用于测试或重新配置）"""
    global _tts_engines
    _tts_engines.clear()
