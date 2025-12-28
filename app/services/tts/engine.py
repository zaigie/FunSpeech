# -*- coding: utf-8 -*-
"""
TTS引擎模块
"""

import sys
import logging
import threading
from typing import List, Dict, Any, Union, Tuple, Optional
from pathlib import Path

from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException
from ...utils.audio import save_audio_array, generate_temp_audio_path

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
    import torch

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


class CosyVoiceTTSEngine:
    """CosyVoice TTS引擎"""

    def __init__(self, load_sft: bool = True, load_clone: bool = True, device: Optional[str] = None):
        """
        初始化TTS引擎

        Args:
            load_sft: 是否加载SFT模型，默认为True
            load_clone: 是否加载零样本克隆模型，默认为True
            device: 指定设备，如 "cuda:0"，None则自动检测
        """
        # 两个不同的模型实例
        self.cosyvoice_sft = None  # 用于预设音色（SFT模式=CosyVoice）
        self.cosyvoice_clone = None  # 用于零样本音色克隆（CosyVoice2 或 CosyVoice3）

        self._device = device if device else self._detect_device()
        self._sft_model_loaded = False
        self._clone_model_loaded = False
        self._clone_model_version = "cosyvoice2"  # 默认版本，加载时会更新
        self._preset_voices = settings.PRESET_VOICES.copy()
        self._voice_manager = None  # 新的音色管理器
        self._load_sft = load_sft  # 保存配置
        self._load_clone = load_clone  # 保存配置
        self._setup_paths()
        self._load_models()

    def _detect_device(self) -> str:
        """检测可用设备"""
        _, device = parse_gpu_config(settings.TTS_GPUS)
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
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3

            fp16_enabled = self._device.startswith("cuda")
            clone_version = settings.CLONE_MODEL_VERSION.lower()

            # 根据配置决定是否加载SFT模型（用于预设音色）
            if self._load_sft:
                try:
                    logger.info(f"正在加载SFT模型（CosyVoice1）到 {self._device}...")
                    self.cosyvoice_sft = CosyVoice(
                        settings.SFT_MODEL_ID,
                        load_jit=True,
                        load_trt=True,
                        fp16=fp16_enabled,
                        device=self._device,  # 传递 device 参数
                    )
                    self._sft_model_loaded = True
                    logger.info(f"SFT模型加载成功，device={self._device}")
                except Exception as e:
                    logger.warning(f"SFT模型加载失败: {str(e)}")
                    self._sft_model_loaded = False
            else:
                logger.info("跳过SFT模型加载（load_sft=False）")
                self._sft_model_loaded = False

            # 加载零样本克隆模型（根据 CLONE_MODEL_VERSION 选择 CosyVoice2 或 CosyVoice3）
            if self._load_clone:
                try:
                    if clone_version == "cosyvoice3":
                        logger.info(f"正在加载零样本克隆模型（CosyVoice3）到 {self._device}...")
                        self.cosyvoice_clone = CosyVoice3(
                            settings.COSYVOICE3_MODEL_ID,
                            load_trt=True,
                            load_vllm=False,
                            fp16=fp16_enabled,
                            device=self._device,
                        )
                        self._clone_model_version = "cosyvoice3"
                    else:
                        logger.info(f"正在加载零样本克隆模型（CosyVoice2）到 {self._device}...")
                        self.cosyvoice_clone = CosyVoice2(
                            settings.CLONE_MODEL_ID,
                            load_jit=True,
                            load_trt=True,
                            load_vllm=False,
                            fp16=fp16_enabled,
                            device=self._device,
                        )
                        self._clone_model_version = "cosyvoice2"
                    self._clone_model_loaded = True
                    logger.info(f"零样本克隆模型（{self._clone_model_version}）加载成功，device={self._device}")
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
        return_timestamps: bool = False,
    ) -> Union[str, Tuple[str, Optional[List[Dict[str, Any]]]]]:
        """语音合成（自动判断音色类型）"""
        try:
            # 检查是否为零样本克隆音色
            if self._voice_manager and self._voice_manager.is_voice_available(voice):
                # 检查是否在零样本克隆音色列表中
                if voice in self._voice_manager.list_clone_voices():
                    sample_rate = 24000
                    logger.debug(f"使用零样本克隆音色模型合成: {voice}")
                    return self._synthesize_with_saved_voice(
                        text,
                        voice,
                        speed,
                        format,
                        sample_rate,
                        volume,
                        prompt,
                        return_timestamps,
                    )

            # 使用预训练音色合成
            return self.synthesize_with_preset_voice(
                text, voice, speed, format, sample_rate, volume, return_timestamps
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
        return_timestamps: bool = False,
    ) -> Union[str, Tuple[str, Optional[List[Dict[str, Any]]]]]:
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
        logger.debug(
            f"使用预训练音色模型合成: {voice}, 格式: {format}, 采样率: {sample_rate}"
        )

        sentences_info = []
        all_audio_segments = []
        current_time = 0.0

        if return_timestamps:
            # 获取CosyVoice的分句结果
            normalized_texts = self.cosyvoice_sft.frontend.text_normalize(
                text, split=True, text_frontend=True
            )
            logger.debug(f"CosyVoice分句结果: {len(normalized_texts)} 个句子")

            # 为每个句子生成音频并记录时间戳
            for sentence_text in normalized_texts:
                sentence_audio_segments = []
                for audio_data in self.cosyvoice_sft.inference_sft(
                    sentence_text, voice, stream=False, speed=speed
                ):
                    sentence_audio_segments.append(audio_data["tts_speech"].numpy())

                if sentence_audio_segments:
                    # 拼接当前句子的音频片段
                    if len(sentence_audio_segments) > 1:
                        import numpy as np

                        sentence_audio = np.concatenate(sentence_audio_segments, axis=1)
                    else:
                        sentence_audio = sentence_audio_segments[0]

                    # 计算当前句子的时长
                    sentence_duration = (
                        sentence_audio.shape[1] / self.cosyvoice_sft.sample_rate
                    )
                    sentence_duration_ms = sentence_duration * 1000

                    # 记录句子信息
                    sentences_info.append(
                        {
                            "text": sentence_text,
                            "begin_time": str(int(current_time)),
                            "end_time": str(int(current_time + sentence_duration_ms)),
                        }
                    )

                    # 添加到总音频
                    all_audio_segments.append(sentence_audio)
                    current_time += sentence_duration_ms

        else:
            # 不需要时间戳，直接合成
            for audio_data in self.cosyvoice_sft.inference_sft(
                text, voice, stream=False, speed=speed
            ):
                all_audio_segments.append(audio_data["tts_speech"].numpy())

        if not all_audio_segments:
            raise DefaultServerErrorException("音频合成失败，未生成任何音频片段")

        # 拼接所有音频片段
        if len(all_audio_segments) > 1:
            import numpy as np

            combined_audio = np.concatenate(all_audio_segments, axis=1)
            logger.debug(f"合并了 {len(all_audio_segments)} 个音频片段")
        else:
            combined_audio = all_audio_segments[0]

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

        if return_timestamps:
            return output_path, sentences_info
        else:
            return output_path

    def _format_prompt_text(self, prompt_text: str) -> str:
        """根据模型版本格式化 prompt_text

        CosyVoice3 需要 'You are a helpful assistant.<|endofprompt|>' 前缀
        """
        if self._clone_model_version == "cosyvoice3":
            if prompt_text and not prompt_text.startswith("You are"):
                return f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
            elif not prompt_text:
                return "You are a helpful assistant.<|endofprompt|>"
        return prompt_text

    def _synthesize_with_saved_voice(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int = 22050,
        volume: int = 50,
        prompt: str = "",
        return_timestamps: bool = False,
    ) -> Union[str, Tuple[str, Optional[List[Dict[str, Any]]]]]:
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
            sentences_info = []
            all_audio_segments = []
            current_time = 0.0

            # 格式化 prompt（CosyVoice3 需要特殊前缀）
            formatted_prompt = self._format_prompt_text(prompt)

            if return_timestamps:
                # 获取CosyVoice的分句结果
                normalized_texts = self.cosyvoice_clone.frontend.text_normalize(
                    text, split=True, text_frontend=True
                )
                logger.debug(f"CosyVoice分句结果: {len(normalized_texts)} 个句子")

                # 为每个句子生成音频并记录时间戳
                for sentence_text in normalized_texts:
                    sentence_audio_segments = []
                    for audio_data in self.cosyvoice_clone.inference_zero_shot(
                        sentence_text,
                        formatted_prompt,  # 使用格式化后的 prompt
                        None,  # 不需要音频
                        zero_shot_spk_id=voice,  # 使用保存的音色ID
                        stream=False,
                        speed=speed,
                    ):
                        sentence_audio_segments.append(audio_data["tts_speech"].numpy())

                    if sentence_audio_segments:
                        # 拼接当前句子的音频片段
                        if len(sentence_audio_segments) > 1:
                            import numpy as np

                            sentence_audio = np.concatenate(
                                sentence_audio_segments, axis=1
                            )
                        else:
                            sentence_audio = sentence_audio_segments[0]

                        # 计算当前句子的时长
                        sentence_duration = (
                            sentence_audio.shape[1] / self.cosyvoice_clone.sample_rate
                        )
                        sentence_duration_ms = sentence_duration * 1000

                        # 记录句子信息
                        sentences_info.append(
                            {
                                "text": sentence_text,
                                "begin_time": str(int(current_time)),
                                "end_time": str(
                                    int(current_time + sentence_duration_ms)
                                ),
                            }
                        )

                        # 添加到总音频
                        all_audio_segments.append(sentence_audio)
                        current_time += sentence_duration_ms

            else:
                # 不需要时间戳，直接合成
                for audio_data in self.cosyvoice_clone.inference_zero_shot(
                    text,
                    formatted_prompt,  # 使用格式化后的 prompt
                    None,  # 不需要音频
                    zero_shot_spk_id=voice,  # 使用保存的音色ID
                    stream=False,
                    speed=speed,
                ):
                    all_audio_segments.append(audio_data["tts_speech"].numpy())

            if not all_audio_segments:
                raise DefaultServerErrorException("音频合成失败，未生成任何音频片段")

            # 拼接所有音频片段
            if len(all_audio_segments) > 1:
                import numpy as np

                combined_audio = np.concatenate(all_audio_segments, axis=1)
                logger.debug(f"合并了 {len(all_audio_segments)} 个音频片段")
            else:
                combined_audio = all_audio_segments[0]

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

            if return_timestamps:
                return output_path, sentences_info
            else:
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
            logger.debug(
                f"音色配置已刷新（cosyvoice1模式），当前有 {len(self._preset_voices)} 个预设音色"
            )
        elif model_mode == "cosyvoice2":
            # 仅使用零样本克隆音色
            if self._voice_manager:
                clone_voices = self._voice_manager.list_clone_voices()
                self._preset_voices = clone_voices.copy()
                logger.debug(
                    f"音色配置已刷新（cosyvoice2模式），当前有 {len(clone_voices)} 个零样本克隆音色"
                )
            else:
                self._preset_voices = []
                logger.debug("音色配置已刷新（cosyvoice2模式），当前没有可用音色")
        else:
            # all模式：包含所有音色
            self._preset_voices = settings.PRESET_VOICES.copy()
            if self._voice_manager:
                # 只从注册表获取零样本克隆音色，避免触发模型重新加载
                clone_voices = self._voice_manager.list_clone_voices()
                for voice in clone_voices:
                    if voice not in self._preset_voices:
                        self._preset_voices.append(voice)
            logger.debug(
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


class MultiGPUTTSEngine:
    """多GPU多副本TTS引擎，支持负载均衡"""

    def __init__(self, load_sft: bool = True, load_clone: bool = True):
        """
        初始化多GPU TTS引擎

        Args:
            load_sft: 是否加载SFT模型
            load_clone: 是否加载零样本克隆模型
        """
        self._load_sft = load_sft
        self._load_clone = load_clone
        self._engines: List[CosyVoiceTTSEngine] = []
        self._devices: List[str] = []
        self._lock = threading.Lock()
        self._current_index = 0
        self._engine_locks: List[threading.Lock] = []  # 每个引擎的锁
        self._engine_active_count: List[int] = []  # 每个引擎当前活跃请求数

        self._init_engines()

    def _init_engines(self):
        """初始化所有GPU上的引擎实例"""
        import torch

        # 解析GPU配置
        gpu_devices, single_device = parse_gpu_config(settings.TTS_GPUS)

        # 单设备模式（包括CPU、单卡GPU、自动检测）
        if len(gpu_devices) == 1:
            logger.info(f"TTS使用单设备模式: {single_device}")
            engine = CosyVoiceTTSEngine(
                load_sft=self._load_sft,
                load_clone=self._load_clone,
                device=single_device
            )
            self._engines.append(engine)
            self._devices.append(engine.device)
            self._engine_locks.append(threading.Lock())
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

            logger.info(f"正在初始化 {device} 上的TTS引擎...")
            try:
                engine = CosyVoiceTTSEngine(
                    load_sft=self._load_sft,
                    load_clone=self._load_clone,
                    device=device
                )
                self._engines.append(engine)
                self._devices.append(device)
                self._engine_locks.append(threading.Lock())
                self._engine_active_count.append(0)
                logger.info(f"{device} 上的TTS引擎初始化成功")
            except Exception as e:
                logger.error(f"{device} 上的TTS引擎初始化失败: {e}")

        if not self._engines:
            raise DefaultServerErrorException("没有成功初始化任何TTS引擎")

        logger.info(f"多GPU TTS引擎初始化完成，共 {len(self._engines)} 个副本")

    def _select_engine(self) -> Tuple[int, CosyVoiceTTSEngine]:
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

    def synthesize_speech(
        self,
        text: str,
        voice: str = "中文女",
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int = 22050,
        volume: int = 50,
        prompt: str = "",
        return_timestamps: bool = False,
    ) -> Union[str, Tuple[str, Optional[List[Dict[str, Any]]]]]:
        """语音合成（负载均衡）"""
        idx, engine = self._select_engine()
        try:
            logger.debug(f"使用引擎 {idx} ({self._devices[idx]}) 处理TTS请求")
            return engine.synthesize_speech(
                text=text,
                voice=voice,
                speed=speed,
                format=format,
                sample_rate=sample_rate,
                volume=volume,
                prompt=prompt,
                return_timestamps=return_timestamps,
            )
        finally:
            self._release_engine(idx)

    def synthesize_with_preset_voice(
        self,
        text: str,
        voice: str = "中文女",
        speed: float = 1.0,
        format: str = "wav",
        sample_rate: int = 22050,
        volume: int = 50,
        return_timestamps: bool = False,
    ) -> Union[str, Tuple[str, Optional[List[Dict[str, Any]]]]]:
        """使用预设音色合成语音（负载均衡）"""
        idx, engine = self._select_engine()
        try:
            logger.debug(f"使用引擎 {idx} ({self._devices[idx]}) 处理预设音色TTS请求")
            return engine.synthesize_with_preset_voice(
                text=text,
                voice=voice,
                speed=speed,
                format=format,
                sample_rate=sample_rate,
                volume=volume,
                return_timestamps=return_timestamps,
            )
        finally:
            self._release_engine(idx)

    def get_voices(self) -> List[str]:
        """获取音色列表（从第一个引擎获取）"""
        if self._engines:
            return self._engines[0].get_voices()
        return []

    def get_voices_info(self) -> Dict[str, Dict[str, Any]]:
        """获取音色详细信息（从第一个引擎获取）"""
        if self._engines:
            return self._engines[0].get_voices_info()
        return {}

    def refresh_voices(self):
        """刷新所有引擎的音色配置"""
        for engine in self._engines:
            engine.refresh_voices()

    def is_sft_model_loaded(self) -> bool:
        """检查SFT模型是否已加载"""
        return any(engine.is_sft_model_loaded() for engine in self._engines)

    def is_clone_model_loaded(self) -> bool:
        """检查零样本克隆模型是否已加载"""
        return any(engine.is_clone_model_loaded() for engine in self._engines)

    def is_tts_model_loaded(self) -> bool:
        """检查TTS模型是否已加载"""
        return any(engine.is_tts_model_loaded() for engine in self._engines)

    @property
    def device(self) -> str:
        """获取设备信息（返回所有设备）"""
        return ",".join(self._devices)

    @property
    def voice_manager(self):
        """获取音色管理器（从第一个引擎获取）"""
        if self._engines:
            return self._engines[0].voice_manager
        return None

    @property
    def cosyvoice_sft(self):
        """获取SFT模型（从第一个引擎获取，兼容旧代码）"""
        if self._engines:
            return self._engines[0].cosyvoice_sft
        return None

    @property
    def cosyvoice_clone(self):
        """获取零样本克隆模型（从第一个引擎获取，兼容旧代码）"""
        if self._engines:
            return self._engines[0].cosyvoice_clone
        return None

    @property
    def _clone_model_version(self):
        """获取克隆模型版本（从第一个引擎获取）"""
        if self._engines:
            return self._engines[0]._clone_model_version
        return "cosyvoice2"

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎状态统计"""
        with self._lock:
            return {
                "total_engines": len(self._engines),
                "devices": self._devices,
                "active_counts": self._engine_active_count.copy(),
            }


# 全局TTS引擎实例
_tts_engine: Optional[MultiGPUTTSEngine] = None
_tts_engine_lock = threading.Lock()


def get_tts_engine() -> MultiGPUTTSEngine:
    """
    获取TTS引擎实例（根据环境变量TTS_MODEL_MODE和TTS_GPUS决定加载策略）

    TTS_GPUS配置:
        - "" 或 "auto": 自动检测设备
        - "cpu": 使用CPU
        - "0": 使用单卡GPU 0
        - "0,1,2": 使用多卡负载均衡

    Returns:
        MultiGPUTTSEngine: TTS引擎实例（支持单设备和多GPU模式）
    """
    global _tts_engine

    with _tts_engine_lock:
        if _tts_engine is None:
            model_mode = settings.TTS_MODEL_MODE.lower()
            load_sft = model_mode in ("all", "cosyvoice1")
            load_clone = model_mode in ("all", "cosyvoice2")

            logger.info(f"创建TTS引擎，TTS_GPUS={settings.TTS_GPUS or '(auto)'}, 模式={model_mode}")
            _tts_engine = MultiGPUTTSEngine(
                load_sft=load_sft,
                load_clone=load_clone
            )
        return _tts_engine


def reset_tts_engines():
    """重置TTS引擎实例（用于测试或重新配置）"""
    global _tts_engine
    _tts_engine = None
