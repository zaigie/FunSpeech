# -*- coding: utf-8 -*-
"""
OpenAI兼容API路由
提供与OpenAI TTS API兼容的接口
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
import logging

from ...core.config import settings
from ...core.exceptions import InvalidParameterException, DefaultServerErrorException
from ...core.security import validate_bearer_token, mask_sensitive_data
from ...models.tts import OpenAITTSRequest
from ...utils.common import generate_task_id, clean_text_for_tts
from ...utils.audio import validate_audio_format
from ...services.tts.engine import get_tts_engine

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/openai/v1/audio", tags=["OpenAI TTS"])


@router.post(
    "/speech",
    response_class=FileResponse,
    summary="OpenAI兼容TTS接口",
    description="完全兼容OpenAI /v1/audio/speech API格式",
)
async def openai_compatible_tts(
    request_body: OpenAITTSRequest, request: Request
) -> FileResponse:
    """兼容OpenAI TTS API的接口"""
    task_id = generate_task_id("openai")
    output_path = None

    try:
        # 验证Bearer Token鉴权
        token = validate_bearer_token(request, task_id)
        logger.debug(
            f"[{task_id}] Bearer Token验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

        logger.info(
            f"[{task_id}] OpenAI兼容接口: 文本='{request_body.input}', 音色={request_body.voice}, 语速={request_body.speed}"
        )

        # 验证format参数
        if request_body.response_format and not validate_audio_format(
            request_body.response_format
        ):
            raise InvalidParameterException(
                f"不支持的音频格式: {request_body.response_format}。支持的格式: {', '.join(settings.SUPPORTED_AUDIO_FORMATS)}",
                task_id,
            )

        # 清理文本
        clean_text = clean_text_for_tts(request_body.input)

        # 获取TTS引擎
        tts_engine = get_tts_engine()

        # 统一语音合成（Engine层自动判断音色类型）
        output_path = tts_engine.synthesize_speech(
            clean_text,
            request_body.voice,
            request_body.speed,
            request_body.response_format,
            22050,  # 默认采样率
            50,  # 默认音量
            request_body.instructions or "",  # 将instructions映射到prompt
        )

        logger.info(f"[{task_id}] OpenAI兼容接口合成完成: {output_path}")

        # 直接返回音频文件
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"speech_{task_id}.wav",
            headers={"task_id": task_id},
        )

    except (InvalidParameterException, DefaultServerErrorException) as e:
        e.task_id = task_id
        logger.error(f"[{task_id}] TTS异常: {e.message}")
        raise HTTPException(status_code=400, detail=e.message)

    except Exception as e:
        logger.error(f"[{task_id}] 未知异常: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部服务错误: {str(e)}")
