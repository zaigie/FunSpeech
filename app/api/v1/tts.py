# -*- coding: utf-8 -*-
"""
TTS API路由
"""

import os
import base64
from fastapi import APIRouter, Request, Form, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import logging

from ...core.config import settings
from ...core.exceptions import (
    InvalidParameterException,
    DefaultServerErrorException,
    UnsupportedSampleRateException,
    AuthenticationException,
)
from ...core.security import (
    validate_token,
    validate_request_appkey,
    mask_sensitive_data,
)
from ...models.tts import (
    TTSResponse,
    TTSHealthCheckResponse,
    VoiceListResponse,
    VoiceDetailResponse,
    VoiceRefreshResponse,
    TTSRequest,
)
from ...models.common import SampleRate, AudioFormat
from ...utils.common import (
    generate_task_id,
    validate_text_input,
    validate_speed_parameter,
    validate_speech_rate_parameter,
    convert_speech_rate_to_speed,
    validate_voice_parameter,
    clean_text_for_tts,
)
from ...utils.audio import (
    validate_reference_audio,
    cleanup_temp_file,
    generate_temp_audio_path,
    validate_audio_format,
    validate_sample_rate,
)
from ...services.tts.engine import get_tts_engine

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/stream/v1/tts", tags=["TTS"])


def save_base64_audio(base64_data: str, task_id: str) -> str:
    """保存base64编码的音频数据为临时文件"""
    try:
        # 解码base64数据
        audio_bytes = base64.b64decode(base64_data)

        # 生成临时文件路径
        temp_path = generate_temp_audio_path(f"ref_{task_id}")

        # 保存文件
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        return temp_path

    except Exception as e:
        raise InvalidParameterException(f"base64音频数据处理失败: {str(e)}")


def format_tts_response(
    task_id: str,
    audio_path: str,
    success: bool = True,
    message: str = "SUCCESS",
) -> dict:
    """格式化TTS响应数据"""
    if success:
        return {
            "task_id": task_id,
            "result": f"/tmp/{os.path.basename(audio_path)}" if audio_path else "",
            "status": 20000000,
            "message": message,
        }
    else:
        return {
            "task_id": task_id,
            "result": "",
            "status": 50000000,
            "message": message,
        }


@router.post(
    "",
    summary="语音合成",
    description="使用指定音色进行文本转语音合成，支持预训练音色和克隆音色。成功时直接返回音频文件二进制数据，失败时返回JSON错误信息",
    responses={
        200: {
            "description": "语音合成成功，返回音频文件",
            "content": {"audio/mpeg": {"schema": {"type": "string"}}},
            "headers": {
                "task_id": {"description": "任务ID", "schema": {"type": "string"}}
            },
        },
        400: {
            "description": "客户端错误",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "任务ID"},
                            "result": {"type": "string", "description": "结果内容"},
                            "status": {"type": "integer", "description": "状态码"},
                            "message": {"type": "string", "description": "错误消息"},
                        },
                    }
                }
            },
        },
        500: {
            "description": "服务端错误",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "任务ID"},
                            "result": {"type": "string", "description": "结果内容"},
                            "status": {"type": "integer", "description": "状态码"},
                            "message": {"type": "string", "description": "错误消息"},
                        },
                    }
                }
            },
        },
    },
    openapi_extra={
        "requestBody": {
            "description": "TTS请求参数",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "appkey": {
                                "type": "string",
                                "description": "应用Appkey，用于API调用认证。如果设置了APPKEY环境变量，则此参数为必需；否则为可选",
                                "example": "your_app_key_here",
                                "minLength": 1,
                                "maxLength": 64,
                            },
                            "text": {
                                "type": "string",
                                "description": "待合成的文本内容",
                                "example": "你好，欢迎使用语音合成服务！",
                                "minLength": 1,
                                "maxLength": 1000,
                            },
                            "voice": {
                                "type": "string",
                                "description": "音色名称",
                                "example": "中文女",
                                "maxLength": 32,
                            },
                            "speech_rate": {
                                "type": "number",
                                "description": "语速倍率，范围-500~500，0为正常语速，负值为减速，正值为加速",
                                "example": 0,
                                "minimum": -500,
                                "maximum": 500,
                            },
                            "volume": {
                                "type": "integer",
                                "description": "音量大小，取值范围0~100，默认值50",
                                "example": 50,
                                "minimum": 0,
                                "maximum": 100,
                                "default": 50,
                            },
                            "format": {
                                "type": "string",
                                "description": f"输出音频格式。支持: {', '.join(AudioFormat.get_enums())}",
                                "example": "wav",
                                "enum": AudioFormat.get_enums(),
                                "default": "wav",
                            },
                            "sample_rate": {
                                "type": "integer",
                                "description": f"音频采样率（Hz）。支持: {', '.join(map(str, SampleRate.get_enums()))}。预设音色默认22050，克隆音色默认24000",
                                "example": 22050,
                                "enum": SampleRate.get_enums(),
                                "default": 22050,
                            },
                            "prompt": {
                                "type": "string",
                                "description": "音色指导文本，用于指导TTS模型的音色生成风格",
                                "example": "说话温柔一些，语气轻松",
                                "maxLength": 500,
                                "default": "",
                            },
                        },
                        "required": ["text"],
                    }
                }
            },
            "required": True,
        }
    },
)
async def synthesize_speech(
    request: Request,
    tts_request: TTSRequest = Body(...),
):
    """语音合成接口，自动识别预设音色和克隆音色"""
    task_id = generate_task_id("tts")
    output_path = None

    try:
        # 验证请求头部（鉴权）
        result, content = validate_token(request, task_id)
        if not result:
            raise AuthenticationException(content, task_id)

        # 验证appkey参数
        result, content = validate_request_appkey(tts_request.appkey, task_id)
        if not result:
            raise AuthenticationException(content, task_id)

        logger.info(
            f"[{task_id}] 开始语音合成: 文本='{tts_request.text}', 音色={tts_request.voice}, 语速={tts_request.speech_rate}, 音量={tts_request.volume}, 格式={tts_request.format}, 采样率={tts_request.sample_rate}"
        )

        # 验证format参数
        if tts_request.format and not validate_audio_format(tts_request.format):
            raise InvalidParameterException(
                f"不支持的音频格式: {tts_request.format}。支持的格式: {', '.join(AudioFormat.get_enums())}",
                task_id,
            )

        # 验证sample_rate参数
        if tts_request.sample_rate and not validate_sample_rate(
            tts_request.sample_rate
        ):
            raise UnsupportedSampleRateException(
                f"不支持的采样率: {tts_request.sample_rate}。支持的采样率: {', '.join(map(str, SampleRate.get_enums()))}",
                task_id,
            )

        # 验证speech_rate参数
        is_valid, message = validate_speech_rate_parameter(tts_request.speech_rate)
        if not is_valid:
            raise InvalidParameterException(message, task_id)

        # 将speech_rate转换为内部speed参数
        speed = convert_speech_rate_to_speed(tts_request.speech_rate)
        logger.info(
            f"[{task_id}] speech_rate={tts_request.speech_rate} 转换为 speed={speed}"
        )

        # 清理文本
        clean_text = clean_text_for_tts(tts_request.text)

        # 获取TTS引擎并合成
        tts_engine = get_tts_engine()
        output_path = tts_engine.synthesize_speech(
            clean_text,
            tts_request.voice,
            speed,
            tts_request.format,
            tts_request.sample_rate,
            tts_request.volume,
            tts_request.prompt or "",
        )

        logger.info(f"[{task_id}] 语音合成完成: {output_path}")

        # 统一使用audio/mpeg作为Content-Type，客户端根据format参数自行保存对应格式
        # 直接返回音频文件
        return FileResponse(
            path=output_path,
            media_type="audio/mpeg",
            filename=f"tts_{task_id}.{tts_request.format}",
            headers={"task_id": task_id},
        )

    except (
        InvalidParameterException,
        DefaultServerErrorException,
        AuthenticationException,
        UnsupportedSampleRateException,
    ) as e:
        e.task_id = task_id
        logger.error(f"[{task_id}] TTS异常: {e.message}")
        response_data = {
            "task_id": task_id,
            "result": "",
            "status": e.status_code,
            "message": e.message,
        }
        return JSONResponse(content=response_data, headers={"task_id": task_id})

    except Exception as e:
        logger.error(f"[{task_id}] 未知异常: {str(e)}")
        response_data = {
            "task_id": task_id,
            "result": "",
            "status": 50000000,
            "message": f"内部服务错误: {str(e)}",
        }
        return JSONResponse(content=response_data, headers={"task_id": task_id})


@router.get(
    "/voices",
    response_model=VoiceListResponse,
    summary="获取音色列表",
    description="返回当前系统中所有可用的音色名称列表",
)
async def get_voice_list(request: Request) -> JSONResponse:
    """获取支持的音色列表"""
    # 鉴权
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "get_voice_list")

    try:
        # 使用懒加载的方式获取音色列表
        from app.core.config import settings

        # 直接返回预设音色，避免触发TTS引擎初始化
        preset_voices = settings.PRESET_VOICES.copy()

        # 尝试从音色管理器的注册表获取克隆音色（不触发模型加载）
        try:
            from app.services.tts.clone import VoiceManager

            voice_manager = VoiceManager()  # 不传入cosyvoice实例，避免模型加载
            clone_voices = voice_manager.list_clone_voices()

            # 合并音色列表
            for voice in clone_voices:
                if voice not in preset_voices:
                    preset_voices.append(voice)
        except Exception as e:
            logger.warning(f"获取克隆音色失败，仅返回预设音色: {e}")

        response_data = {"voices": preset_voices, "total": len(preset_voices)}
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"获取音色列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取音色列表失败: {str(e)}")


@router.get(
    "/voices/info",
    response_model=VoiceDetailResponse,
    summary="获取详细音色信息",
    description="返回所有音色的详细信息",
)
async def get_voice_info(request: Request) -> JSONResponse:
    """获取详细的音色信息"""
    # 鉴权
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "get_voice_info")

    try:
        tts_engine = get_tts_engine()
        voices_info = tts_engine.get_voices_info()

        response_data = {
            "voices": voices_info,
            "total": len(voices_info),
            "preset_count": len(
                [v for v in voices_info.values() if v["type"] == "preset"]
            ),
            "clone_count": len(
                [v for v in voices_info.values() if v["type"] == "clone"]
            ),
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"获取音色信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取音色信息失败: {str(e)}")


@router.post(
    "/voices/refresh",
    response_model=VoiceRefreshResponse,
    summary="刷新音色配置",
    description="重新扫描并加载音色配置",
)
async def refresh_voices(request: Request) -> JSONResponse:
    """刷新音色配置"""
    # 鉴权
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "refresh_voices")

    try:
        tts_engine = get_tts_engine()
        tts_engine.refresh_voices()

        voices = tts_engine.get_voices()
        response_data = {
            "message": "音色配置已刷新",
            "voices": voices,
            "total": len(voices),
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"刷新音色配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"刷新音色配置失败: {str(e)}")


@router.get(
    "/health",
    response_model=TTSHealthCheckResponse,
    summary="TTS服务健康检查",
    description="检查文本转语音服务的运行状态",
)
async def health_check(request: Request) -> JSONResponse:
    """TTS服务健康检查"""
    # 鉴权
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "health_check")

    try:
        tts_engine = get_tts_engine()

        response_data = {
            "status": "healthy",
            "sft_model_loaded": tts_engine.is_sft_model_loaded(),
            "tts_model_loaded": tts_engine.is_tts_model_loaded(),
            "device": tts_engine.device,
            "preset_voices": tts_engine.get_voices(),
            "version": settings.APP_VERSION,
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "sft_model_loaded": False,
                "tts_model_loaded": False,
                "device": "unknown",
                "preset_voices": [],
                "version": settings.APP_VERSION,
            }
        )
