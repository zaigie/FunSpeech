#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunSpeech API Server 启动脚本
简化的启动方式，适用于开发和测试
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from main import settings, logger
    import uvicorn

    print("=" * 60)
    print("🚀 FunSpeech API Server")
    print("=" * 60)
    print(f"📍 服务地址: http://{settings.HOST}:{settings.PORT}")
    print(f"🔧 设备配置: {settings.DEVICE}")
    print(f"🔧 TTS设备配置: {settings.TTS_DEVICE}")
    print(f"🧠 ASR模型模式: {settings.ASR_MODEL_MODE}")
    print(f"🧠 TTS模型模式: {settings.TTS_MODEL_MODE}")
    print(
        f"📖 API文档: http://{settings.HOST}:{settings.PORT}/docs"
        if settings.DEBUG
        else "📖 API文档: 禁用"
    )
    print(
        f"🩺 ASR健康检查: http://{settings.HOST}:{settings.PORT}/stream/v1/asr/health"
    )
    print(
        f"🩺 TTS健康检查: http://{settings.HOST}:{settings.PORT}/stream/v1/tts/health"
    )
    print("=" * 60)

    # 预加载所有模型
    try:
        from app.utils.model_loader import preload_models, print_model_statistics

        preload_result = preload_models()

        # 打印详细的加载统计到控制台
        print()
        print_model_statistics(preload_result, use_logger=False)
        print()

    except Exception as e:
        print(f"\n❌ 模型预加载失败: {e}")
        print("⚠️  服务将继续启动，模型将在首次使用时加载\n")

    print("=" * 60)
    print("🌐 正在启动API服务器...")
    print("=" * 60)

    try:
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            log_level="debug" if settings.DEBUG else settings.LOG_LEVEL.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)
