# -*- coding: utf-8 -*-
"""API v1版本路由"""

from fastapi import APIRouter
from .asr import router as asr_router
from .tts import router as tts_router
from .openai import router as openai_router
from .websocket_tts import router as websocket_tts_router
from .websocket_asr import router as websocket_asr_router
from .async_tts import router as async_tts_router

api_router = APIRouter()

api_router.include_router(asr_router)
api_router.include_router(tts_router)
api_router.include_router(openai_router)
api_router.include_router(websocket_tts_router)
api_router.include_router(websocket_asr_router)
api_router.include_router(async_tts_router)
