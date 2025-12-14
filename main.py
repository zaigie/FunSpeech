# -*- coding: utf-8 -*-
"""
FunSpeech API 主程序
"""

import uvicorn
import logging

from app.main import create_app
from app.core.config import settings

# 创建应用实例
app = create_app()

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("启动FunSpeech API服务器")
    logger.info(
        f"服务器配置: Host={settings.HOST}, Port={settings.PORT}, Debug={settings.DEBUG}"
    )
    logger.info(f"GPU配置: ASR_GPUS={settings.ASR_GPUS or '(auto)'}, TTS_GPUS={settings.TTS_GPUS or '(auto)'}")

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
