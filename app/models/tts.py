# -*- coding: utf-8 -*-
"""
TTS数据模型
定义语音合成相关的请求和响应模型
"""

from typing import Optional, List, Any, Union, Dict
from pydantic import BaseModel, Field, field_validator
from enum import Enum

from .common import (
    AudioFormat,
    SampleRate,
    BaseResponse,
    HealthCheckResponse,
    ErrorResponse,
)
from ..core.config import settings


class TTSModelType(str, Enum):
    """TTS模型类型"""

    TTS_1 = "tts-1"
    TTS_1_HD = "tts-1-hd"
    COSYVOICE = "cosyvoice"


class VoiceType(str, Enum):
    """音色类型"""

    PRESET = "preset"
    CLONE = "clone"


# ============= 请求模型 =============


class TTSRequest(BaseModel):
    """预设音色TTS请求模型"""

    appkey: Optional[str] = Field(
        None,
        description="应用Appkey，用于API调用认证。如果设置了APPKEY环境变量，则此参数为必需；否则为可选",
        example="your_app_key_here",
        min_length=1,
        max_length=64,
    )

    text: str = Field(
        ...,
        description="待合成的文本内容",
        example="你好，欢迎使用语音合成服务！",
        min_length=1,
        max_length=1000,
    )

    voice: str = Field(
        "中文女",
        description="音色名称",
        example="中文女",
        max_length=32,
    )

    speech_rate: float = Field(
        0,
        description="语速倍率，范围-500~500，0为正常语速，负值为减速，正值为加速",
        example=0,
        ge=-500,
        le=500,
    )

    volume: int = Field(
        50,
        description="音量大小，取值范围0~100，默认值50",
        example=50,
        ge=0,
        le=100,
    )

    format: Optional[AudioFormat] = Field(
        "wav",
        description=f"输出音频格式。支持: {', '.join(AudioFormat.get_enums())}",
        example="wav",
    )

    sample_rate: Optional[SampleRate] = Field(
        22050,
        description=f"音频采样率（Hz）。支持: {', '.join(map(str, SampleRate.get_enums()))}。预设音色默认22050，克隆音色默认24000",
        example=22050,
    )

    prompt: Optional[str] = Field(
        "",
        description="音色指导文本，用于指导TTS模型的音色生成风格",
        example="说话温柔一些，语气轻松",
        max_length=500,
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("文本内容不能为空")
        if len(v) > settings.MAX_TEXT_LENGTH:
            raise ValueError(
                f"文本长度超过限制，最大支持{settings.MAX_TEXT_LENGTH}个字符"
            )
        return v.strip()

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("音色参数不能为空")
        return v.strip()


class OpenAITTSRequest(BaseModel):
    """OpenAI兼容TTS请求模型"""

    model: TTSModelType = Field(
        TTSModelType.TTS_1,
        description="使用的模型",
        example="tts-1",
    )

    input: str = Field(
        ...,
        description="待合成的文本",
        example="Hello, this is a test of the speech synthesis system.",
        min_length=1,
        max_length=1000,
    )

    voice: str = Field(
        ...,
        description="音色名称或参考音频文件路径",
        example="中文女",
        min_length=1,
        max_length=256,
    )

    response_format: AudioFormat = Field(
        AudioFormat.WAV,
        description="响应音频格式",
        example="wav",
    )

    speed: float = Field(
        1.0,
        description="语速倍率",
        example=1.0,
        ge=0.5,
        le=2.0,
    )

    instructions: Optional[str] = Field(
        None,
        description="音色指导文本，用于指导TTS模型的音色生成风格（等同于prompt参数）",
        example="说话温柔一些，语气轻松",
        max_length=500,
    )


# ============= 响应模型 =============


class TTSSuccessResponse(BaseResponse):
    """TTS成功响应模型"""

    result: str = Field(
        ...,
        description="生成音频文件的下载链接",
        example="/tmp/tts_cf7b0c5339244ee29cd4e43fb97f1234.wav",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "tts_cf7b0c5339244ee29cd4e43fb97f1234",
                "result": "/tmp/tts_cf7b0c5339244ee29cd4e43fb97f1234.wav",
                "status": 20000000,
                "message": "SUCCESS",
            }
        }


class TTSErrorResponse(ErrorResponse):
    """TTS错误响应模型"""

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "tts_8bae3613dfc54ebfa811a17d8a7a1234",
                "result": "",
                "status": 40000004,
                "message": "INVALID_VOICE",
            }
        }


class TTSHealthCheckResponse(HealthCheckResponse):
    """TTS健康检查响应模型"""

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "sft_model_loaded": True,
                "tts_model_loaded": True,
                "device": "cuda:0",
                "preset_voices": ["中文女", "中文男", "英文女", "英文男"],
                "version": "1.0.0",
                "message": "TTS service is running normally",
                "memory_usage": {
                    "gpu_memory_used": "2.1GB",
                    "gpu_memory_total": "8.0GB",
                    "cpu_memory_used": "1.5GB",
                },
                "model_info": {
                    "sft_model": "CosyVoice-300M-SFT",
                    "tts_model": "CosyVoice-300M",
                },
            }
        },
    }

    sft_model_loaded: bool = Field(..., description="SFT模型是否已加载")
    tts_model_loaded: bool = Field(..., description="TTS模型是否已加载")
    device: str = Field(..., description="推理设备")
    preset_voices: List[str] = Field(..., description="可用的预设音色列表")
    memory_usage: Optional[dict] = Field(None, description="内存使用情况")
    model_info: Optional[dict] = Field(None, description="模型信息")


# ============= 音色相关模型 =============


class VoiceInfo(BaseModel):
    """音色信息模型"""

    name: str = Field(..., description="音色名称")
    type: VoiceType = Field(..., description="音色类型")
    language: str = Field(..., description="支持的语言")
    gender: Optional[str] = Field(None, description="性别")
    description: Optional[str] = Field(None, description="音色描述")
    sample_rate: Optional[int] = Field(None, description="音频采样率")
    available: bool = Field(True, description="是否可用")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "中文女",
                "type": "preset",
                "language": "zh-CN",
                "gender": "female",
                "description": "标准中文女声",
                "sample_rate": 22050,
                "available": True,
            }
        }


class VoiceListResponse(BaseModel):
    """音色列表响应模型"""

    voices: List[str] = Field(..., description="音色名称列表")
    total: int = Field(..., description="音色总数")

    class Config:
        json_schema_extra = {
            "example": {
                "voices": ["中文女", "中文男", "活力女孩"],
                "total": 4,
            }
        }


class VoiceDetailResponse(BaseModel):
    """音色详细信息响应模型"""

    voices: Dict[str, VoiceInfo] = Field(..., description="音色详细信息字典")
    total: int = Field(..., description="音色总数")
    preset_count: int = Field(..., description="预设音色数量")
    clone_count: int = Field(..., description="克隆音色数量")

    class Config:
        json_schema_extra = {
            "example": {
                "voices": {
                    "中文女": {
                        "name": "中文女",
                        "type": "preset",
                        "language": "zh-CN",
                        "gender": "female",
                        "description": "标准中文女声",
                        "sample_rate": 22050,
                        "available": True,
                    }
                },
                "total": 4,
                "preset_count": 2,
                "clone_count": 2,
            }
        }


class VoiceRefreshResponse(BaseModel):
    """音色刷新响应模型"""

    message: str = Field(..., description="刷新结果消息")
    voices: List[str] = Field(..., description="刷新后的音色列表")
    total: int = Field(..., description="刷新后的音色总数")
    added: Optional[List[str]] = Field(None, description="新增的音色")
    removed: Optional[List[str]] = Field(None, description="移除的音色")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "特征音色配置已刷新",
                "voices": ["中文女", "中文男", "活力女孩"],
                "total": 4,
                "added": ["新音色1"],
                "removed": ["旧音色1"],
            }
        }


# ============= 联合响应类型 =============

TTSResponse = Union[TTSSuccessResponse, TTSErrorResponse]
