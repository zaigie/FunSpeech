# -*- coding: utf-8 -*-
"""
工具模块
包含通用工具函数和辅助功能
"""

from .common import *
from .audio import *
from .text_processing import apply_itn_to_text

__all__ = [
    # 通用工具函数
    "generate_task_id",
    "validate_text_input",
    "validate_speed_parameter",
    "validate_speech_rate_parameter",
    "convert_speech_rate_to_speed",
    "validate_voice_parameter",
    "clean_text_for_tts",
    "parse_language_code",
    "estimate_synthesis_time",
    # 音频工具函数
    "save_audio_array",
    "load_audio_file",
    "generate_temp_audio_path",
    "cleanup_temp_file",
    # ITN（逆文本标准化）功能 - 基于WeText
    "apply_itn_to_text",
]
