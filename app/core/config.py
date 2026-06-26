# -*- coding: utf-8 -*-
"""网关进程的统一配置

模型加载相关 env(MODELSCOPE_PATH / VAD_MODEL / SFT_MODEL_ID / TTS_LOAD_TRT 等)
全部转移到对应的 services/* 子服务,本进程只持有网关与子服务之间的协作配置。
"""

import os
from pathlib import Path
from typing import List, Optional


class Settings:
    """网关进程配置类"""

    # 应用信息
    APP_NAME: str = "FunSpeech API Server"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "微服务化语音网关 (ASR + TTS),兼容阿里云/OpenAI API"

    # 服务器
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # 鉴权
    APPTOKEN: Optional[str] = None
    APPKEY: Optional[str] = None

    # 路径
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: str = "temp"
    DATA_DIR: str = "data"

    # 日志
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = BASE_DIR / "logs" / "funspeech.log"
    LOG_MAX_BYTES: int = 20 * 1024 * 1024
    LOG_BACKUP_COUNT: int = 50

    # ASR — 网关需要知道的: model 配置入口、模式、自定义预热列表
    ASR_MODELS_CONFIG: str = BASE_DIR / "app/services/asr/models.json"
    ASR_MODEL_MODE: str = "all"  # all / offline / realtime, 子服务也读这个 env
    ASR_ENABLE_REALTIME_PUNC: bool = False  # 转发给 funasr 子服务的实时 PUNC 开关
    AUTO_LOAD_CUSTOM_ASR_MODELS: str = ""  # 网关启动时预热的额外模型 id, 逗号分隔

    # 流式 ASR 远场过滤(网关侧句子状态机用,与子服务无关)
    ASR_ENABLE_NEARFIELD_FILTER: bool = True
    ASR_NEARFIELD_RMS_THRESHOLD: float = 0.01
    ASR_NEARFIELD_FILTER_LOG_ENABLED: bool = True

    # TTS — 网关只选择后端; 具体模型形态由对应子服务 env 控制。
    # 可选:
    #   cosyvoice                 legacy CosyVoice SFT/2/3 子服务
    #   qwen3-tts                 legacy qwen-tts Python 包子服务
    #   cosyvoice3-vllm-omni      CosyVoice3 via vLLM-Omni
    #   qwen3-tts-vllm-omni       Qwen3-TTS via vLLM-Omni
    TTS_ENGINE: str = "cosyvoice"
    # legacy cosyvoice 用 all/sft/clone; qwen3 可用 all/base/custom/voicedesign
    # 做 get_voices 过滤。模型加载模式不要再从网关推断, 以子服务 env 为准。
    TTS_MODEL_MODE: str = "all"

    # 微服务 — 子服务 URL / 鉴权 / 超时
    INTERNAL_SERVICE_TOKEN: Optional[str] = None
    SERVICE_REQUEST_TIMEOUT: float = 60.0
    SERVICE_HEALTHCHECK_INTERVAL: float = 5.0
    FUNASR_SERVICE_URLS: str = ""
    DOLPHIN_SERVICE_URLS: str = ""
    QWEN3_ASR_SERVICE_URLS: str = ""
    COSYVOICE_SERVICE_URLS: str = ""
    QWEN3_TTS_SERVICE_URLS: str = ""
    COSYVOICE3_VLLM_OMNI_SERVICE_URLS: str = ""
    QWEN3_TTS_VLLM_OMNI_SERVICE_URLS: str = ""

    # 音频处理
    MAX_AUDIO_SIZE: int = 100 * 1024 * 1024
    MAX_TEXT_LENGTH: int = 1000

    # TTS 预设音色(用于 get_voices_info 返回展示)
    PRESET_VOICES: List[str] = [
        "中文女",
        "中文男",
        "日语男",
        "粤语女",
        "英文女",
        "英文男",
        "韩语女",
    ]

    # TTS 参数限制
    MIN_SPEED: float = 0.5
    MAX_SPEED: float = 2.0
    DEFAULT_SPEED: float = 1.0
    MIN_SPEECH_RATE: int = -500
    MAX_SPEECH_RATE: int = 500
    DEFAULT_SPEECH_RATE: int = 0

    # 参考音频(网关上传校验用)
    MAX_REFERENCE_AUDIO_DURATION: int = 30
    MIN_REFERENCE_AUDIO_DURATION: float = 1.0

    def __init__(self):
        self._load_from_env()
        self._ensure_directories()

    def _load_from_env(self):
        # 服务器
        self.HOST = os.getenv("HOST", self.HOST)
        self.PORT = int(os.getenv("PORT", str(self.PORT)))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # 日志
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FILE = os.getenv("LOG_FILE", self.LOG_FILE)
        self.LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(self.LOG_MAX_BYTES)))
        self.LOG_BACKUP_COUNT = int(
            os.getenv("LOG_BACKUP_COUNT", str(self.LOG_BACKUP_COUNT))
        )

        # 鉴权 — 空字符串视为未配置 (docker compose 的 ${APPTOKEN:-} 默认塞空串)
        self.APPTOKEN = os.getenv("APPTOKEN", self.APPTOKEN) or None
        self.APPKEY = os.getenv("APPKEY", self.APPKEY) or None

        # ASR
        self.ASR_MODEL_MODE = os.getenv("ASR_MODEL_MODE", self.ASR_MODEL_MODE)
        self.ASR_ENABLE_REALTIME_PUNC = (
            os.getenv("ASR_ENABLE_REALTIME_PUNC", "false").lower() == "true"
        )
        self.AUTO_LOAD_CUSTOM_ASR_MODELS = os.getenv(
            "AUTO_LOAD_CUSTOM_ASR_MODELS", self.AUTO_LOAD_CUSTOM_ASR_MODELS
        )

        # TTS
        self.TTS_ENGINE = os.getenv("TTS_ENGINE", self.TTS_ENGINE)
        self.TTS_MODEL_MODE = os.getenv("TTS_MODEL_MODE", self.TTS_MODEL_MODE)

        # 微服务
        self.INTERNAL_SERVICE_TOKEN = os.getenv(
            "INTERNAL_SERVICE_TOKEN", self.INTERNAL_SERVICE_TOKEN
        )
        self.SERVICE_REQUEST_TIMEOUT = float(
            os.getenv("SERVICE_REQUEST_TIMEOUT", str(self.SERVICE_REQUEST_TIMEOUT))
        )
        self.SERVICE_HEALTHCHECK_INTERVAL = float(
            os.getenv(
                "SERVICE_HEALTHCHECK_INTERVAL", str(self.SERVICE_HEALTHCHECK_INTERVAL)
            )
        )
        self.FUNASR_SERVICE_URLS = os.getenv(
            "FUNASR_SERVICE_URLS", self.FUNASR_SERVICE_URLS
        )
        self.DOLPHIN_SERVICE_URLS = os.getenv(
            "DOLPHIN_SERVICE_URLS", self.DOLPHIN_SERVICE_URLS
        )
        self.QWEN3_ASR_SERVICE_URLS = os.getenv(
            "QWEN3_ASR_SERVICE_URLS", self.QWEN3_ASR_SERVICE_URLS
        )
        self.COSYVOICE_SERVICE_URLS = os.getenv(
            "COSYVOICE_SERVICE_URLS", self.COSYVOICE_SERVICE_URLS
        )
        self.QWEN3_TTS_SERVICE_URLS = os.getenv(
            "QWEN3_TTS_SERVICE_URLS", self.QWEN3_TTS_SERVICE_URLS
        )
        self.COSYVOICE3_VLLM_OMNI_SERVICE_URLS = os.getenv(
            "COSYVOICE3_VLLM_OMNI_SERVICE_URLS",
            self.COSYVOICE3_VLLM_OMNI_SERVICE_URLS,
        )
        self.QWEN3_TTS_VLLM_OMNI_SERVICE_URLS = os.getenv(
            "QWEN3_TTS_VLLM_OMNI_SERVICE_URLS",
            self.QWEN3_TTS_VLLM_OMNI_SERVICE_URLS,
        )

        # 远场过滤
        self.ASR_ENABLE_NEARFIELD_FILTER = (
            os.getenv("ASR_ENABLE_NEARFIELD_FILTER", "true").lower() == "true"
        )
        self.ASR_NEARFIELD_RMS_THRESHOLD = float(
            os.getenv(
                "ASR_NEARFIELD_RMS_THRESHOLD", str(self.ASR_NEARFIELD_RMS_THRESHOLD)
            )
        )
        self.ASR_NEARFIELD_FILTER_LOG_ENABLED = (
            os.getenv("ASR_NEARFIELD_FILTER_LOG_ENABLED", "true").lower() == "true"
        )

    def _ensure_directories(self):
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

    @property
    def models_config_path(self) -> str:
        return str(self.BASE_DIR / self.ASR_MODELS_CONFIG)

    @property
    def docs_url(self) -> Optional[str]:
        return "/docs" if self.DEBUG else None

    @property
    def redoc_url(self) -> Optional[str]:
        return "/redoc" if self.DEBUG else None


settings = Settings()
