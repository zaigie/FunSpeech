# -*- coding: utf-8 -*-
"""
OpenAI兼容API路由
提供与OpenAI TTS API兼容的接口
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import logging

from ...core.config import settings
from ...core.executor import run_sync
from ...core.exceptions import InvalidParameterException, DefaultServerErrorException
from ...core.security import validate_bearer_token, mask_sensitive_data
from ...models.tts import OpenAITTSRequest
from ...models.common import AudioFormat
from ...utils.common import generate_task_id, clean_text_for_tts
from ...utils.audio import validate_audio_format
from ...services.tts.engine import get_tts_engine

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/openai/v1/audio", tags=["OpenAI TTS"])


@router.post(
    "/speech",
    summary="OpenAI兼容TTS接口",
    description="完全兼容OpenAI /v1/audio/speech API格式。成功时返回音频文件，失败时返回JSON错误信息",
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
)
async def openai_compatible_tts(request_body: OpenAITTSRequest, request: Request):
    """兼容OpenAI TTS API的接口"""
    task_id = generate_task_id("openai")
    output_path = None

    try:
        # 验证Bearer Token鉴权
        token = validate_bearer_token(request, task_id)
        logger.debug(
            f"[{task_id}] Bearer Token验证通过, token: {mask_sensitive_data(token) if token != 'optional' else 'optional'}"
        )

        logger.debug(
            f"[{task_id}] OpenAI兼容接口: 文本='{request_body.input}', 音色={request_body.voice}, 语速={request_body.speed}"
        )

        # 验证format参数
        if request_body.response_format and not validate_audio_format(
            request_body.response_format
        ):
            raise InvalidParameterException(
                f"不支持的音频格式: {request_body.response_format}。支持的格式: {', '.join(AudioFormat.get_enums())}",
                task_id,
            )

        # 清理文本
        clean_text = clean_text_for_tts(request_body.input)

        # 获取TTS引擎
        tts_engine = get_tts_engine()

        sample_rate = 22050  # 默认采样率

        # 统一语音合成（使用线程池执行，避免阻塞事件循环）
        output_path = await run_sync(
            tts_engine.synthesize_speech,
            clean_text,
            request_body.voice,
            request_body.speed,
            request_body.response_format,
            sample_rate,
            50,  # 默认音量
            request_body.instructions or "",  # 将instructions映射到prompt
        )

        logger.debug(f"[{task_id}] OpenAI兼容接口合成完成: {output_path}")

        # 统一使用audio/mpeg作为Content-Type，客户端根据response_format参数自行保存对应格式
        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename=f"speech_{task_id}.{request_body.response_format}",
            headers={"task_id": task_id},
        )

    except (InvalidParameterException, DefaultServerErrorException) as e:
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
