# -*- coding: utf-8 -*-
"""
ASR数据模型
定义语音识别相关的请求和响应模型
"""

from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field, field_validator

from .common import (
    AudioFormat,
    SampleRate,
    BaseResponse,
    HealthCheckResponse,
    ErrorResponse,
)


# ============= 请求模型 =============


class ASRQueryParams(BaseModel):
    """ASR接口查询参数模型"""

    appkey: Optional[str] = Field(
        None,
        description="应用Appkey，用于API调用认证。如果设置了APPKEY环境变量，则此参数为必需；否则为可选",
        example="your_app_key_here",
        min_length=1,
        max_length=64,
    )

    format: Optional[AudioFormat] = Field(
        "pcm",
        description=f"音频格式。支持: {', '.join(AudioFormat.get_enums())}。仅在使用audio_address参数时生效，使用二进制音频流时默认为wav格式",
        example="pcm",
    )

    sample_rate: Optional[SampleRate] = Field(
        16000,
        description=f"音频采样率（Hz）。支持: {', '.join(map(str, SampleRate.get_enums()))}",
        example=16000,
    )

    vocabulary_id: Optional[str] = Field(
        None,
        description="热词表ID，用于提高特定词汇的识别准确率",
        example="vocab_12345",
        max_length=32,
    )

    customization_id: Optional[str] = Field(
        "paraformer-large",
        description="自定义模型ID，指定使用的ASR模型",
        example="paraformer-large",
        max_length=64,
    )

    enable_punctuation_prediction: Optional[bool] = Field(
        False,
        description="是否启用标点符号预测",
        example=True,
    )

    enable_inverse_text_normalization: Optional[bool] = Field(
        False,
        description="是否启用反向文本标准化（将中文数字转为阿拉伯数字）",
        example=False,
    )

    enable_voice_detection: Optional[bool] = Field(
        False,
        description="是否启用语音活动检测（VAD）",
        example=False,
    )

    disfluency: Optional[bool] = Field(
        False,
        description="是否过滤语气词（嗯、啊等）",
        example=False,
    )

    audio_address: Optional[str] = Field(
        None,
        description="音频文件下载链接（HTTP/HTTPS）",
        example="https://example.com/audio.wav",
        max_length=512,
    )

    dolphin_lang_sym: Optional[str] = Field(
        "zh",
        description="Dolphin引擎语言符号",
        example="zh",
        max_length=8,
    )

    dolphin_region_sym: Optional[str] = Field(
        "SHANGHAI",
        description="Dolphin引擎区域符号",
        example="SHANGHAI",
        max_length=16,
    )


class ASRHeaders(BaseModel):
    """ASR接口请求头模型"""

    x_nls_token: Optional[str] = Field(
        None,
        alias="X-NLS-Token",
        description="访问令牌，用于身份认证",
        example="your_access_token_here",
        min_length=10,
        max_length=256,
    )

    content_type: Optional[str] = Field(
        "application/octet-stream",
        alias="Content-Type",
        description="请求体内容类型",
        example="application/octet-stream",
    )


# ============= 响应模型 =============


class ASRSuccessResponse(BaseResponse):
    """ASR成功响应模型"""

    result: str = Field(
        ...,
        description="识别结果文本",
        example="北京的天气怎么样？",
        max_length=10000,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "cf7b0c5339244ee29cd4e43fb97f1234",
                "result": "北京的天气怎么样？",
                "status": 20000000,
                "message": "SUCCESS",
            }
        }


class ASRErrorResponse(ErrorResponse):
    """ASR错误响应模型"""

    result: str = Field("", description="识别结果（错误时为空）")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "8bae3613dfc54ebfa811a17d8a7a1234",
                "result": "",
                "status": 40000001,
                "message": "Gateway:ACCESS_DENIED:The token 'invalid_token' is invalid!",
            }
        }


class ASRHealthCheckResponse(HealthCheckResponse):
    """ASR健康检查响应模型"""

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "device": "cuda:0",
                "version": "1.0.0",
                "message": "ASR service is running normally",
                "loaded_models": ["paraformer-large", "sensevoice-small"],
                "memory_usage": {
                    "gpu_memory_used": "2.1GB",
                    "gpu_memory_total": "8.0GB",
                },
                "asr_model_mode": "realtime",
            },
        },
    }

    model_loaded: bool = Field(..., description="模型是否已加载")
    device: str = Field(..., description="推理设备")
    loaded_models: Optional[List[str]] = Field([], description="已加载的模型列表")
    memory_usage: Optional[dict] = Field(None, description="内存使用情况")
    asr_model_mode: Optional[str] = Field(None, description="当前ASR模型加载模式")


# ============= 模型相关 =============


class ASRModelInfo(BaseModel):
    """新的ASR模型信息模型，支持离线和实时模型分离"""

    id: str = Field(..., description="模型id")
    name: str = Field(..., description="模型名称")
    engine: str = Field(..., description="引擎类型", example="funasr")
    description: str = Field(..., description="模型描述")
    languages: List[str] = Field(..., description="支持的语言列表")
    default: bool = Field(False, description="是否为默认模型")
    loaded: bool = Field(False, description="是否已加载")
    supports_realtime: bool = Field(False, description="是否支持实时识别")
    offline_model: Optional[dict] = Field(None, description="离线模型信息")
    realtime_model: Optional[dict] = Field(None, description="实时模型信息")
    asr_model_mode: str = Field(..., description="当前ASR模型加载模式")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "paraformer-large",
                "name": "Paraformer Large",
                "engine": "funasr",
                "description": "高精度中文语音识别模型",
                "languages": ["zh"],
                "default": True,
                "loaded": True,
                "supports_realtime": True,
                "offline_model": {
                    "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    "exists": True,
                },
                "realtime_model": {
                    "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                    "exists": True,
                },
                "asr_model_mode": "realtime",
            }
        }


class ASRModelsResponse(BaseModel):
    """ASR模型列表响应模型"""

    models: List[ASRModelInfo] = Field(..., description="模型列表")
    total: int = Field(..., description="模型总数")
    loaded_count: int = Field(..., description="已加载模型数量")
    asr_model_mode: str = Field(..., description="当前ASR模型加载模式")

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "id": "paraformer-large",
                        "name": "Paraformer Large",
                        "engine": "funasr",
                        "description": "高精度中文语音识别模型",
                        "languages": ["zh"],
                        "default": True,
                        "loaded": True,
                        "supports_realtime": True,
                        "offline_model": {
                            "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                            "exists": True,
                        },
                        "realtime_model": {
                            "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                            "exists": True,
                        },
                        "asr_model_mode": "realtime",
                    }
                ],
                "total": 3,
                "loaded_count": 1,
                "asr_model_mode": "realtime",
            }
        }


# ============= 联合响应类型 =============

ASRResponse = Union[ASRSuccessResponse, ASRErrorResponse]
