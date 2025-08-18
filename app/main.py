# -*- coding: utf-8 -*-
"""
FastAPI应用创建和配置
"""

import warnings
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .core.exceptions import (
    APIException,
    api_exception_handler,
    general_exception_handler,
)
from .core.logging import setup_logging
from .api.v1 import api_router

# 忽略 Pydantic V2 兼容性警告
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
warnings.filterwarnings("ignore", message=".*has conflict with protected namespace.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def create_app() -> FastAPI:
    """创建FastAPI应用"""

    # 设置日志
    setup_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
    )

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册异常处理器
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # 注册静态文件服务（用于TTS生成的音频文件）
    app.mount("/tmp", StaticFiles(directory=settings.TEMP_DIR), name="temp_files")

    # 注册API路由
    app.include_router(api_router)

    # 根路径
    @app.get("/", summary="根路径", description="API服务根路径")
    async def root():
        return {
            "message": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": settings.APP_DESCRIPTION,
            "endpoints": {
                "asr": "/stream/v1/asr",
                "asr_models": "/stream/v1/asr/models",
                "asr_health": "/stream/v1/asr/health",
                "tts": "/stream/v1/tts",
                "tts_clone": "/stream/v1/tts/clone",
                "tts_clone_eq": "/stream/v1/tts/clone_eq",
                "tts_openai": "/openai/v1/audio/speech",
                "tts_voices": "/stream/v1/tts/voices",
                "tts_health": "/stream/v1/tts/health",
                "docs": settings.docs_url or "禁用",
            },
        }

    return app
