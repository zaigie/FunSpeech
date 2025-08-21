# -*- coding: utf-8 -*-
"""
通用数据模型
定义通用的枚举、基础模型等
"""

from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class AudioFormat(str, Enum):
    """支持的音频格式"""

    PCM = "pcm"
    WAV = "wav"
    OPUS = "opus"
    SPEEX = "speex"
    AMR = "amr"
    MP3 = "mp3"
    AAC = "aac"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"


class SampleRate(int, Enum):
    """支持的采样率"""

    RATE_8000 = 8000
    RATE_16000 = 16000
    RATE_22050 = 22050
    RATE_44100 = 44100
    RATE_48000 = 48000


class BaseResponse(BaseModel):
    """基础响应模型"""

    task_id: str = Field(..., description="任务ID")
    status: int = Field(..., description="状态码")
    message: str = Field(..., description="响应消息")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""

    status: str = Field(..., description="服务状态", example="healthy")
    version: str = Field(..., description="服务版本", example="1.0.0")
    message: str = Field(
        ..., description="状态消息", example="Service is running normally"
    )


class ErrorResponse(BaseResponse):
    """错误响应模型"""

    result: str = Field("", description="结果内容")
