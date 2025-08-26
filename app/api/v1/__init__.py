# -*- coding: utf-8 -*-
"""
API v1版本路由
"""

from fastapi import APIRouter
from .asr import router as asr_router
from .tts import router as tts_router
from .openai import router as openai_router
from .websocket_tts import router as websocket_tts_router

# 创建v1版本的主路由器
api_router = APIRouter()

# 注册子路由
api_router.include_router(asr_router)
api_router.include_router(tts_router)
api_router.include_router(openai_router)
api_router.include_router(websocket_tts_router)
