# -*- coding: utf-8 -*-
"""
统一配置管理
整合ASR和TTS的配置选项
"""

import os
from typing import Optional, List
from pathlib import Path


class Settings:
    """统一应用配置类"""

    # 应用信息
    APP_NAME: str = "FunSpeech API Server"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "基于FunASR的语音识别和CosyVoice的语音合成API服务"

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # 鉴权配置
    APPTOKEN: Optional[str] = None  # 从环境变量APPTOKEN读取，如果为None则鉴权可选
    APPKEY: Optional[str] = None  # 从环境变量APPKEY读取，如果为None则appkey可选

    # 设备配置
    DEVICE: str = "auto"  # auto, cpu, cuda:0, npu:0
    TTS_DEVICE: str = "auto"  # auto, cpu, cuda:0

    # 路径配置
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: str = "temp"
    DATA_DIR: str = "data"  # 数据持久化目录
    MODELSCOPE_PATH: str = os.path.expanduser("~/.cache/modelscope/hub")

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = BASE_DIR / "logs" / "funspeech.log"
    LOG_MAX_BYTES: int = 20 * 1024 * 1024  # 20MB
    LOG_BACKUP_COUNT: int = 50  # 保留50个备份文件

    # ASR模型配置
    FUNASR_AUTOMODEL_KWARGS = {
        "trust_remote_code": False,
        "disable_update": True,
        "disable_pbar": True,
        "disable_log": True,  # 禁用FunASR的tables输出
    }
    ASR_MODELS_CONFIG: str = BASE_DIR / "app/services/asr/models.json"
    ASR_MODEL_MODE: str = "all"  # ASR模型加载模式: realtime, offline, all
    ASR_ENABLE_REALTIME_PUNC: bool = False  # 是否启用实时标点模型（用于中间结果展示）
    VAD_MODEL: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    VAD_MODEL_REVISION: str = "v2.0.4"
    PUNC_MODEL: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    PUNC_MODEL_REVISION: str = "v2.0.4"
    PUNC_REALTIME_MODEL: str = (
        "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    )
    SPK_MODEL: Optional[str] = None  # 禁用说话人分离功能

    # TTS模型配置
    SFT_MODEL_ID: str = "iic/CosyVoice-300M-SFT"  # 预训练音色模型（CosyVoice）
    CLONE_MODEL_ID: str = "iic/CosyVoice2-0.5B"  # 零样本克隆模型（CosyVoice2）
    TTS_MODEL_MODE: str = "all"  # TTS模型加载模式: all, cosyvoice1, cosyvoice2

    # 音频处理配置
    MAX_AUDIO_SIZE: int = 100 * 1024 * 1024  # 100MB
    MAX_TEXT_LENGTH: int = 1000  # 最大文本长度

    # TTS预设音色列表
    PRESET_VOICES: List[str] = [
        "中文女",
        "中文男",
        "日语男",
        "粤语女",
        "英文女",
        "英文男",
        "韩语女",
    ]

    # TTS参数限制
    MIN_SPEED: float = 0.5
    MAX_SPEED: float = 2.0
    DEFAULT_SPEED: float = 1.0

    # 阿里云speech_rate参数限制
    MIN_SPEECH_RATE: int = -500
    MAX_SPEECH_RATE: int = 500
    DEFAULT_SPEECH_RATE: int = 0

    # 参考音频配置
    MAX_REFERENCE_AUDIO_DURATION: int = 30  # 秒
    MIN_REFERENCE_AUDIO_DURATION: float = 1.0  # 秒

    def __init__(self):
        """从环境变量读取配置"""
        self._load_from_env()
        self._ensure_directories()

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 服务器配置
        self.HOST = os.getenv("HOST", self.HOST)
        self.PORT = int(os.getenv("PORT", str(self.PORT)))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # 日志配置
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FILE = os.getenv("LOG_FILE", self.LOG_FILE)
        self.LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(self.LOG_MAX_BYTES)))
        self.LOG_BACKUP_COUNT = int(
            os.getenv("LOG_BACKUP_COUNT", str(self.LOG_BACKUP_COUNT))
        )

        # 鉴权配置
        self.APPTOKEN = os.getenv("APPTOKEN", self.APPTOKEN)
        self.APPKEY = os.getenv("APPKEY", self.APPKEY)

        # 设备配置
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        self.TTS_DEVICE = os.getenv("TTS_DEVICE", self.TTS_DEVICE)

        # ASR模型配置
        self.ASR_MODEL_MODE = os.getenv("ASR_MODEL_MODE", self.ASR_MODEL_MODE)
        self.ASR_ENABLE_REALTIME_PUNC = (
            os.getenv("ASR_ENABLE_REALTIME_PUNC", "false").lower() == "true"
        )

        # TTS模型配置
        self.TTS_MODEL_MODE = os.getenv("TTS_MODEL_MODE", self.TTS_MODEL_MODE)

    def _ensure_directories(self):
        """确保必需的目录存在"""
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

    @property
    def models_config_path(self) -> str:
        """获取模型配置文件的完整路径"""
        return str(self.BASE_DIR / self.ASR_MODELS_CONFIG)

    @property
    def docs_url(self) -> Optional[str]:
        """获取文档URL"""
        return "/docs" if self.DEBUG else None

    @property
    def redoc_url(self) -> Optional[str]:
        """获取ReDoc URL"""
        return "/redoc" if self.DEBUG else None


# 全局配置实例
settings = Settings()
