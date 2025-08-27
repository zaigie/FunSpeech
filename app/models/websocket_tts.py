# -*- coding: utf-8 -*-
"""
WebSocket TTS 流式语音合成相关数据模型 - 阿里云协议
"""

from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field, field_validator
import uuid
from .common import SampleRate


class AliyunWSHeader(BaseModel):
    """阿里云WebSocket消息头部"""

    message_id: str = Field(..., description="消息ID")
    task_id: str = Field(..., description="任务ID")
    namespace: str = Field(..., description="命名空间")
    name: str = Field(..., description="消息名称")
    appkey: Optional[str] = Field(None, description="应用密钥")
    status: Optional[int] = Field(None, description="状态码")
    status_text: Optional[str] = Field(None, description="状态消息")
    status_message: Optional[str] = Field(None, description="状态消息")

    @staticmethod
    def generate_message_id() -> str:
        """生成32位消息ID"""
        return str(uuid.uuid4()).replace("-", "")[:32]


class AliyunStartSynthesisPayload(BaseModel):
    """StartSynthesis 消息负载"""

    voice: str = Field(default="中文女", description="音色")
    format: str = Field(default="PCM", description="音频格式")
    sample_rate: int = Field(default=22050, description="采样率")
    volume: int = Field(default=50, ge=0, le=100, description="音量")
    speech_rate: int = Field(default=0, ge=-500, le=500, description="语速")
    pitch_rate: int = Field(default=0, ge=-500, le=500, description="音调")
    enable_subtitle: bool = Field(default=False, description="启用字幕")
    platform: str = Field(default="python", description="平台")
    method: int = Field(default=1, description="合成方法")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        supported_formats = ["PCM", "WAV", "MP3"]
        if v.upper() not in supported_formats:
            raise ValueError(f"不支持的音频格式: {v}")
        return v.upper()

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v):
        supported_rates = SampleRate.get_enums()
        if v not in supported_rates:
            raise ValueError(f"不支持的采样率: {v}")
        return v


class AliyunRunSynthesisPayload(BaseModel):
    """RunSynthesis 消息负载"""

    text: str = Field(..., min_length=1, max_length=1000, description="待合成的文本")


class AliyunSubtitle(BaseModel):
    """字幕信息"""

    text: str = Field("", description="文本内容")
    begin_time: int = Field(0, description="开始时间")
    end_time: int = Field(0, description="结束时间")
    begin_index: int = Field(0, description="开始索引")
    end_index: int = Field(0, description="结束索引")
    sentence: bool = Field(False, description="是否为句子")
    phoneme_list: List[Dict[str, Any]] = Field(
        default_factory=list, description="音素列表"
    )


class AliyunSynthesisPayload(BaseModel):
    """合成结果负载"""

    session_id: Optional[str] = Field(None, description="会话ID")
    index: Optional[int] = Field(None, description="索引")
    subtitles: Optional[List[AliyunSubtitle]] = Field(None, description="字幕信息")


class AliyunWSMessage(BaseModel):
    """阿里云WebSocket消息"""

    header: AliyunWSHeader = Field(..., description="消息头部")
    payload: Optional[
        Union[
            AliyunStartSynthesisPayload,
            AliyunRunSynthesisPayload,
            AliyunSynthesisPayload,
        ]
    ] = Field(None, description="消息负载")


class AliyunTTSNamespace:
    """阿里云TTS命名空间常量"""

    FLOWING_SPEECH_SYNTHESIZER = "FlowingSpeechSynthesizer"
    SPEECH_SYNTHESIZER = "SpeechSynthesizer"
    SPEECH_LONG_SYNTHESIZER = "SpeechLongSynthesizer"
    DEFAULT = "Default"


class AliyunTTSMessageName:
    """阿里云TTS消息名称常量"""

    # 客户端发送
    START_SYNTHESIS = "StartSynthesis"
    RUN_SYNTHESIS = "RunSynthesis"
    STOP_SYNTHESIS = "StopSynthesis"

    # 服务端响应
    SYNTHESIS_STARTED = "SynthesisStarted"
    SENTENCE_BEGIN = "SentenceBegin"
    SENTENCE_SYNTHESIS = "SentenceSynthesis"
    SENTENCE_END = "SentenceEnd"
    SYNTHESIS_COMPLETED = "SynthesisCompleted"
    TASK_FAILED = "TaskFailed"


class AliyunTTSStatus:
    """阿里云TTS状态码常量"""

    SUCCESS = 20000000  # 成功
    TASK_FAILED = 40000000  # 任务失败
    INVALID_PARAMETER = 40000001  # 参数无效
    AUTHENTICATION_FAILED = 40100005  # 认证失败
    QUOTA_EXCEEDED = 40300016  # 配额超限
    INTERNAL_ERROR = 50000000  # 内部错误
    SERVICE_UNAVAILABLE = 50300018  # 服务不可用
