# -*- coding: utf-8 -*-
"""
异步TTS数据模型
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

from .common import AudioFormat, SampleRate, BaseResponse


class AsyncTTSPayload(BaseModel):
    """异步TTS载荷"""

    tts_request: 'AsyncTTSRequestData' = Field(..., description="TTS请求数据")
    enable_notify: bool = Field(False, description="是否启用回调通知")
    notify_url: Optional[str] = Field(None, description="回调通知URL")


class AsyncTTSRequestData(BaseModel):
    """异步TTS请求数据"""

    voice: str = Field(..., description="音色名称", example="xiaoyun")
    sample_rate: int = Field(16000, description="音频采样率", example=16000)
    format: str = Field("wav", description="音频格式", example="wav")
    text: str = Field(..., description="待合成文本", min_length=1, max_length=5000)
    enable_subtitle: bool = Field(True, description="是否启用字幕", example=True)

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v not in [8000, 16000, 22050, 24000, 48000]:
            raise ValueError(f"不支持的采样率: {v}")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ["pcm", "wav", "mp3"]:
            raise ValueError(f"不支持的音频格式: {v}")
        return v


class AsyncTTSContext(BaseModel):
    """异步TTS上下文"""

    device_id: Optional[str] = Field(None, description="设备ID", example="my_device_id")


class AsyncTTSHeader(BaseModel):
    """异步TTS请求头"""

    appkey: str = Field(..., description="应用Appkey", example="yourAppkey")
    token: str = Field(..., description="访问令牌", example="yourToken")


class AsyncTTSRequest(BaseModel):
    """异步TTS完整请求"""

    payload: AsyncTTSPayload = Field(..., description="请求载荷")
    context: Optional[AsyncTTSContext] = Field(None, description="请求上下文")
    header: AsyncTTSHeader = Field(..., description="请求头")


class AsyncTTSTaskData(BaseModel):
    """异步TTS任务响应数据"""

    task_id: str = Field(..., description="任务ID")
    audio_address: Optional[str] = Field(None, description="音频下载地址")
    notify_custom: Optional[str] = Field(None, description="自定义通知数据（与notify_url相同）")
    sentences: Optional[List[Dict[str, Any]]] = Field(None, description="句子时间戳信息")


class AsyncTTSResponse(BaseModel):
    """异步TTS响应"""

    status: int = Field(..., description="状态码")
    error_code: int = Field(..., description="错误码")
    error_message: str = Field(..., description="错误消息")
    request_id: str = Field(..., description="请求ID")
    data: Optional[AsyncTTSTaskData] = Field(None, description="响应数据")


class AsyncTTSErrorResponse(BaseModel):
    """异步TTS错误响应"""

    error_message: str = Field(..., description="错误消息")
    error_code: int = Field(..., description="错误码")
    request_id: str = Field(..., description="请求ID")
    url: str = Field(..., description="请求URL")
    status: int = Field(..., description="HTTP状态码")


class SentenceInfo(BaseModel):
    """句子信息"""

    text: str = Field(..., description="句子文本")
    begin_time: str = Field(..., description="开始时间(毫秒)")
    end_time: str = Field(..., description="结束时间(毫秒)")