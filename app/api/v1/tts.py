# -*- coding: utf-8 -*-
"""
TTS API路由
"""

import os
import base64
from fastapi import APIRouter, Request, Form, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from ...core.config import settings
from ...core.exceptions import (
    TTSException,
    TTSModelException,
    InvalidVoiceException,
    InvalidSpeedException,
    InvalidSpeechRateException,
    ReferenceAudioException,
)
from ...core.security import validate_xls_token, mask_sensitive_data
from ...models.tts import (
    TTSResponse,
    TTSHealthCheckResponse,
    VoiceListResponse,
    VoiceDetailResponse,
    VoiceRefreshResponse,
    TTSSuccessResponse,
    TTSErrorResponse,
    PresetVoiceTTSRequest,
)
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
        raise ReferenceAudioException(f"base64音频数据处理失败: {str(e)}")


def format_tts_response(
    task_id: str, audio_path: str, success: bool = True, message: str = "SUCCESS"
) -> dict:
    """格式化TTS响应数据"""
    if success:
        return {
            "task_id": task_id,
            "audio_url": f"/tmp/{os.path.basename(audio_path)}" if audio_path else "",
            "status": 20000000,
            "message": message,
        }
    else:
        return {
            "task_id": task_id,
            "audio_url": "",
            "status": 50000000,
            "message": message,
        }


@router.post(
    "",
    response_model=TTSResponse,
    summary="语音合成",
    description="使用指定音色进行文本转语音合成，支持预训练音色和克隆音色",
)
async def synthesize_speech(
    request: Request,
    tts_request: PresetVoiceTTSRequest = Body(...),
) -> JSONResponse:
    """语音合成接口，自动识别预设音色和克隆音色"""
    task_id = generate_task_id("tts")
    output_path = None

    try:
        # 验证请求头部（鉴权）
        token = validate_xls_token(request, task_id)
        logger.info(
            f"[{task_id}] 请求验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

        logger.info(
            f"[{task_id}] 开始语音合成: 文本='{tts_request.text}', 音色={tts_request.voice}, 语速={tts_request.speech_rate}"
        )

        # 验证speech_rate参数
        is_valid, message = validate_speech_rate_parameter(tts_request.speech_rate)
        if not is_valid:
            raise InvalidSpeechRateException(message, task_id)

        # 将speech_rate转换为内部speed参数
        speed = convert_speech_rate_to_speed(tts_request.speech_rate)
        logger.info(
            f"[{task_id}] speech_rate={tts_request.speech_rate} 转换为 speed={speed}"
        )

        # 清理文本
        clean_text = clean_text_for_tts(tts_request.text)

        # 获取TTS引擎并合成（Engine层会自动判断音色类型）
        tts_engine = get_tts_engine()
        output_path = tts_engine.synthesize_speech(clean_text, tts_request.voice, speed)

        logger.info(f"[{task_id}] 语音合成完成: {output_path}")

        # 返回成功响应
        response_data = format_tts_response(task_id, output_path, True, "SUCCESS")
        return JSONResponse(content=response_data, headers={"task_id": task_id})

    except TTSException as e:
        e.task_id = task_id
        logger.error(f"[{task_id}] TTS异常: {e.message}")
        response_data = format_tts_response(task_id, "", False, e.message)
        return JSONResponse(content=response_data, headers={"task_id": task_id})

    except Exception as e:
        logger.error(f"[{task_id}] 未知异常: {str(e)}")
        response_data = format_tts_response(
            task_id, "", False, f"内部服务错误: {str(e)}"
        )
        return JSONResponse(content=response_data, headers={"task_id": task_id})


@router.get(
    "/voices",
    response_model=VoiceListResponse,
    summary="获取音色列表",
    description="返回当前系统中所有可用的音色名称列表",
)
async def get_voice_list(request: Request) -> JSONResponse:
    """获取支持的音色列表"""
    try:
        # 验证请求头部（鉴权）
        token = validate_xls_token(request)
        logger.info(
            f"音色列表请求验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

        tts_engine = get_tts_engine()
        voices = tts_engine.get_voices()

        response_data = {"voices": voices, "total": len(voices)}

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
    try:
        # 验证请求头部（鉴权）
        token = validate_xls_token(request)
        logger.info(
            f"音色信息请求验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

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
    try:
        # 验证请求头部（鉴权）
        token = validate_xls_token(request)
        logger.info(
            f"音色刷新请求验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

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
    try:
        # 验证请求头部（鉴权）
        token = validate_xls_token(request)
        logger.info(
            f"健康检查请求验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

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
