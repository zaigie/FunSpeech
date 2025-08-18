# -*- coding: utf-8 -*-
"""
统一异常处理模块
定义所有自定义异常类和错误处理函数
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class APIException(Exception):
    """API基础异常类"""

    def __init__(
        self, status_code: int, message: str, task_id: str = "", error_code: str = ""
    ):
        self.status_code = status_code
        self.message = message
        self.task_id = task_id
        self.error_code = error_code or self._get_error_code(status_code)
        super().__init__(self.message)

    def _get_error_code(self, status_code: int) -> str:
        """根据状态码获取错误代码"""
        code_mapping = {
            40000001: "AUTHENTICATION_FAILED",
            40000002: "INVALID_MESSAGE",
            40000003: "INVALID_PARAMETER",
            40000004: "INVALID_VOICE",
            40000005: "INVALID_SPEED",
            40000006: "REFERENCE_AUDIO_ERROR",
            40000011: "MISSING_APPKEY",
            40000012: "INVALID_APPKEY",
            40000013: "PARAMETER_ERROR",
            40000014: "UNSUPPORTED_AUDIO_FORMAT",
            40000015: "UNSUPPORTED_SAMPLE_RATE",
            40000021: "EMPTY_AUDIO_DATA",
            40000022: "INVALID_AUDIO_FORMAT",
            40000023: "AUDIO_FILE_TOO_LARGE",
            40000024: "AUDIO_DOWNLOAD_FAILED",
            41010101: "UNSUPPORTED_SAMPLE_RATE",
            50000000: "INTERNAL_SERVER_ERROR",
            50000001: "MODEL_ERROR",
            50000002: "AUDIO_PROCESSING_ERROR",
        }
        return code_mapping.get(status_code, "UNKNOWN_ERROR")


# ASR相关异常
class ASRException(APIException):
    """ASR异常基类"""

    pass


class AuthenticationException(ASRException):
    """身份认证异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000001, message, task_id)


class InvalidParameterException(ASRException):
    """无效参数异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000003, message, task_id)


class InvalidMessageException(ASRException):
    """无效消息异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000002, message, task_id)


class UnsupportedSampleRateException(ASRException):
    """不支持的采样率异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(41010101, message, task_id)


class AudioProcessingException(ASRException):
    """音频处理异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(50000002, message, task_id)


# TTS相关异常
class TTSException(APIException):
    """TTS异常基类"""

    pass


class TTSModelException(TTSException):
    """TTS模型异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(50000001, message, task_id)


class InvalidVoiceException(TTSException):
    """无效音色异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000004, message, task_id)


class InvalidSpeedException(TTSException):
    """无效语速异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000005, message, task_id)


class InvalidSpeechRateException(TTSException):
    """无效speech_rate异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000007, message, task_id)


class ReferenceAudioException(TTSException):
    """参考音频异常"""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(40000006, message, task_id)


# 异常处理器
async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """API异常处理器"""
    logger.error(f"[{exc.task_id}] API异常: {exc.message}")

    response_data = {
        "task_id": exc.task_id,
        "result": "",
        "status": exc.status_code,
        "message": exc.message,
        "error_code": exc.error_code,
    }

    # 为TTS接口添加额外字段
    if isinstance(exc, TTSException):
        response_data.update({"audio_url": "", "error_details": exc.message})

    return JSONResponse(
        content=response_data,
        headers={"task_id": exc.task_id} if exc.task_id else {},
        status_code=400 if exc.status_code >= 40000000 else 500,
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)

    response_data = {
        "task_id": "",
        "result": "",
        "status": 50000000,
        "message": f"内部服务错误: {str(exc)}",
        "error_code": "INTERNAL_SERVER_ERROR",
    }

    return JSONResponse(content=response_data, status_code=500)


# 为了向后兼容，保留旧的异常名称
FunASRException = ASRException
funasr_exception_handler = api_exception_handler
