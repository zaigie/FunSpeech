# -*- coding: utf-8 -*-
"""
异步TTS API路由
"""

import os
import logging
import threading
import time
import httpx
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import uuid

from ...core.config import settings
from ...core.database import db_manager
from ...core.exceptions import (
    InvalidParameterException,
    DefaultServerErrorException,
    AuthenticationException,
)
from ...core.security import validate_token, validate_request_appkey
from ...models.async_tts import (
    AsyncTTSRequest,
    AsyncTTSResponse,
    AsyncTTSErrorResponse,
    AsyncTTSTaskData,
    SentenceInfo,
)
from ...utils.common import generate_task_id, clean_text_for_tts
from ...services.tts.engine import get_tts_engine

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/rest/v1/tts", tags=["Async TTS"])

# 后台任务处理线程标志
_background_worker_started = False
_worker_lock = threading.Lock()


import asyncio


async def _send_notify_callback(notify_url: str, response_data: dict, is_error: bool = False):
    """发送回调通知"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                notify_url,
                json=response_data,
                headers={'Content-Type': 'application/json'}
            )
            logger.info(f"回调通知发送成功: {notify_url}, 状态码: {response.status_code}")
            return True
    except Exception as e:
        logger.error(f"发送回调通知失败: {notify_url}, 错误: {str(e)}")
        return False


def _send_notify_sync(notify_url: str, response_data: dict, is_error: bool = False):
    """同步版本的回调通知发送"""
    try:
        # 在新的事件循环中运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_send_notify_callback(notify_url, response_data, is_error))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"发送回调通知异常: {notify_url}, 错误: {str(e)}")
        return False


def _process_async_tasks():
    """后台处理异步TTS任务"""
    logger.info("异步TTS后台处理线程启动")

    while True:
        try:
            # 获取待处理任务
            pending_tasks = db_manager.get_pending_tasks(limit=5)

            for task in pending_tasks:
                try:
                    task_id = task['task_id']
                    logger.info(f"处理异步TTS任务: {task_id}")

                    # 获取TTS引擎
                    tts_engine = get_tts_engine()

                    # 清理文本
                    clean_text = clean_text_for_tts(task['text'])

                    # 合成语音并获取句子时间戳
                    result = tts_engine.synthesize_speech(
                        clean_text,
                        task['voice'],
                        1.0,  # 默认语速
                        task['format'],
                        task['sample_rate'],
                        50,   # 默认音量
                        "",   # 默认prompt
                        task['enable_subtitle']  # 返回时间戳
                    )

                    # 解析返回结果
                    if task['enable_subtitle']:
                        output_path, sentences = result
                    else:
                        output_path = result
                        sentences = None

                    # 生成访问URL
                    audio_filename = os.path.basename(output_path)
                    audio_address = f"/tmp/{audio_filename}"

                    # 更新任务状态为成功
                    db_manager.update_task_status(
                        task_id,
                        'SUCCESS',
                        audio_address=audio_address,
                        sentences=sentences,
                        error_code=20000000,
                        error_message='SUCCESS'
                    )

                    logger.info(f"异步TTS任务完成: {task_id}")

                    # 发送成功回调通知
                    if task.get('enable_notify') and task.get('notify_url'):
                        success_response = AsyncTTSResponse(
                            status=200,
                            error_code=20000000,
                            error_message="SUCCESS",
                            request_id=str(uuid.uuid4()).replace('-', ''),
                            data=AsyncTTSTaskData(
                                task_id=task_id,
                                audio_address=audio_address,
                                notify_custom=task['notify_url'],
                                sentences=sentences
                            )
                        )
                        _send_notify_sync(task['notify_url'], success_response.model_dump())

                except Exception as e:
                    logger.error(f"处理异步TTS任务失败: {task['task_id']}, 错误: {str(e)}")
                    db_manager.update_task_status(
                        task['task_id'],
                        'FAILED',
                        error_code=50000000,
                        error_message=str(e)
                    )

                    # 发送失败回调通知
                    if task.get('enable_notify') and task.get('notify_url'):
                        error_response = AsyncTTSErrorResponse(
                            error_message=str(e),
                            error_code=50000000,
                            request_id=str(uuid.uuid4()).replace('-', ''),
                            url=task['notify_url'],
                            status=500
                        )
                        _send_notify_sync(task['notify_url'], error_response.model_dump(), is_error=True)

            # 清理旧任务
            db_manager.cleanup_old_tasks()

            # 等待一段时间再处理下一批
            time.sleep(2)

        except Exception as e:
            logger.error(f"异步TTS后台处理异常: {str(e)}")
            time.sleep(5)


def _start_background_worker():
    """启动后台工作线程"""
    global _background_worker_started

    with _worker_lock:
        if not _background_worker_started:
            worker_thread = threading.Thread(target=_process_async_tasks, daemon=True)
            worker_thread.start()
            _background_worker_started = True
            logger.info("异步TTS后台工作线程已启动")


@router.post(
    "/async",
    summary="提交异步语音合成任务",
    description="提交长文本异步语音合成任务，返回任务ID用于后续查询结果",
    responses={
        200: {
            "description": "任务提交成功",
            "content": {
                "application/json": {
                    "schema": AsyncTTSResponse.model_json_schema()
                }
            }
        },
        400: {
            "description": "客户端错误",
            "content": {
                "application/json": {
                    "schema": AsyncTTSErrorResponse.model_json_schema()
                }
            }
        }
    }
)
async def submit_async_tts(request: Request, tts_request: AsyncTTSRequest):
    """提交异步TTS任务"""
    # 启动后台处理线程（如果未启动）
    _start_background_worker()

    request_id = str(uuid.uuid4()).replace('-', '')
    task_id = str(uuid.uuid4()).replace('-', '')

    try:
        # 验证header中的token和appkey
        if not tts_request.header.token:
            raise AuthenticationException("缺少访问令牌", task_id)

        if not tts_request.header.appkey:
            raise AuthenticationException("缺少应用密钥", task_id)

        # 验证请求参数
        payload = tts_request.payload.tts_request

        if len(payload.text) > 5000:
            raise InvalidParameterException("文本长度超过限制，最大支持5000个字符", task_id)

        if not payload.voice.strip():
            raise InvalidParameterException("音色参数不能为空", task_id)

        # 验证回调设置
        if tts_request.payload.enable_notify:
            if not tts_request.payload.notify_url:
                raise DefaultServerErrorException("启用回调通知时必须设置notify_url", task_id)
            if not tts_request.payload.notify_url.startswith(('http://', 'https://')):
                raise InvalidParameterException("notify_url必须是有效的HTTP/HTTPS URL", task_id)

        logger.info(f"提交异步TTS任务: {task_id}, 文本长度: {len(payload.text)}")

        # 创建任务记录
        task_data = {
            'task_id': task_id,
            'request_id': request_id,
            'text': payload.text,
            'voice': payload.voice,
            'sample_rate': payload.sample_rate,
            'format': payload.format,
            'enable_subtitle': payload.enable_subtitle,
            'enable_notify': tts_request.payload.enable_notify,
            'notify_url': tts_request.payload.notify_url if tts_request.payload.enable_notify else None,
        }

        success = db_manager.create_task(task_data)
        if not success:
            raise DefaultServerErrorException("创建任务失败", task_id)

        # 返回成功响应
        response_data = AsyncTTSResponse(
            status=200,
            error_code=20000000,
            error_message="SUCCESS",
            request_id=request_id,
            data=AsyncTTSTaskData(task_id=task_id)
        )

        return JSONResponse(content=response_data.model_dump())

    except (InvalidParameterException, AuthenticationException, DefaultServerErrorException) as e:
        logger.error(f"异步TTS提交失败: {str(e)}")
        error_response = AsyncTTSErrorResponse(
            error_message=str(e),
            error_code=getattr(e, 'status_code', 40000000),
            request_id=request_id,
            url="/rest/v1/tts/async",
            status=400 if isinstance(e, (InvalidParameterException, AuthenticationException)) else 500
        )
        return JSONResponse(
            content=error_response.model_dump(),
            status_code=error_response.status
        )

    except Exception as e:
        logger.error(f"异步TTS未知异常: {str(e)}")
        error_response = AsyncTTSErrorResponse(
            error_message=f"内部服务错误: {str(e)}",
            error_code=50000000,
            request_id=request_id,
            url="/rest/v1/tts/async",
            status=500
        )
        return JSONResponse(content=error_response.model_dump(), status_code=500)


@router.get(
    "/async",
    summary="获取异步语音合成结果",
    description="根据任务ID获取异步语音合成的结果状态",
    responses={
        200: {
            "description": "查询成功",
            "content": {
                "application/json": {
                    "schema": AsyncTTSResponse.model_json_schema()
                }
            }
        },
        400: {
            "description": "客户端错误",
            "content": {
                "application/json": {
                    "schema": AsyncTTSErrorResponse.model_json_schema()
                }
            }
        }
    }
)
async def get_async_tts_result(
    request: Request,
    appkey: str = Query(..., description="应用Appkey"),
    token: str = Query(..., description="访问令牌"),
    task_id: str = Query(..., description="任务ID")
):
    """获取异步TTS结果"""
    request_id = str(uuid.uuid4()).replace('-', '')

    try:
        # 验证参数
        if not token:
            raise AuthenticationException("缺少访问令牌", task_id)

        if not appkey:
            raise AuthenticationException("缺少应用密钥", task_id)

        if not task_id:
            raise InvalidParameterException("缺少任务ID", task_id)

        # 获取任务信息
        task = db_manager.get_task(task_id)
        if not task:
            raise InvalidParameterException("任务不存在", task_id)

        logger.info(f"查询异步TTS任务: {task_id}, 状态: {task['status']}")

        # 构建响应数据
        data = AsyncTTSTaskData(
            task_id=task_id,
            audio_address=task.get('audio_address'),
            notify_custom=task.get('notify_url') if task.get('enable_notify') else None
        )

        # 如果任务完成且启用了字幕，添加句子信息
        if task['status'] == 'SUCCESS' and task.get('sentences'):
            data.sentences = task['sentences']

        response_data = AsyncTTSResponse(
            status=200,
            error_code=task['error_code'],
            error_message=task['error_message'],
            request_id=request_id,
            data=data
        )

        return JSONResponse(content=response_data.model_dump())

    except (InvalidParameterException, AuthenticationException) as e:
        logger.error(f"查询异步TTS失败: {str(e)}")
        error_response = AsyncTTSErrorResponse(
            error_message=str(e),
            error_code=getattr(e, 'status_code', 40000000),
            request_id=request_id,
            url="/rest/v1/tts/async",
            status=400
        )
        return JSONResponse(content=error_response.model_dump(), status_code=400)

    except Exception as e:
        logger.error(f"查询异步TTS未知异常: {str(e)}")
        error_response = AsyncTTSErrorResponse(
            error_message=f"内部服务错误: {str(e)}",
            error_code=50000000,
            request_id=request_id,
            url="/rest/v1/tts/async",
            status=500
        )
        return JSONResponse(content=error_response.model_dump(), status_code=500)