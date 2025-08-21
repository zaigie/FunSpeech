# -*- coding: utf-8 -*-
"""
通用工具函数
包含任务ID生成、参数验证等通用功能
"""

import uuid
import hashlib
import time
import re
from typing import Optional, Tuple
from ..core.config import settings


def generate_task_id(prefix: str = "") -> str:
    """生成唯一的任务ID

    Args:
        prefix: 任务ID前缀

    Returns:
        生成的任务ID
    """
    timestamp = str(int(time.time() * 1000))
    random_id = str(uuid.uuid4()).replace("-", "")
    combined = timestamp + random_id

    # 使用MD5哈希生成32位字符串
    task_id = hashlib.md5(combined.encode()).hexdigest()

    if prefix:
        return f"{prefix}_{task_id}"
    return task_id


def validate_text_input(text: str) -> Tuple[bool, str]:
    """验证输入文本

    Args:
        text: 待合成的文本

    Returns:
        (is_valid, message): 验证结果和消息
    """
    if not text or not text.strip():
        return False, "文本内容不能为空"

    text = text.strip()

    if len(text) > settings.MAX_TEXT_LENGTH:
        return False, f"文本长度超过限制，最大支持{settings.MAX_TEXT_LENGTH}个字符"

    # 检查是否包含有效字符
    if not re.search(r"[\u4e00-\u9fff\w\s]", text):
        return False, "文本内容无效，请输入有效的中文、英文或数字"

    return True, "验证通过"


def validate_speed_parameter(speed: float) -> Tuple[bool, str]:
    """验证语速参数

    Args:
        speed: 语速值

    Returns:
        (is_valid, message): 验证结果和消息
    """
    if speed < settings.MIN_SPEED:
        return False, f"语速过慢，最小值为{settings.MIN_SPEED}"

    if speed > settings.MAX_SPEED:
        return False, f"语速过快，最大值为{settings.MAX_SPEED}"

    return True, "验证通过"


def validate_speech_rate_parameter(speech_rate: float) -> Tuple[bool, str]:
    """验证阿里云speech_rate参数

    Args:
        speech_rate: 语速值 (-500~500)

    Returns:
        (is_valid, message): 验证结果和消息
    """
    if speech_rate < settings.MIN_SPEECH_RATE:
        return False, f"语速过慢，最小值为{settings.MIN_SPEECH_RATE}"

    if speech_rate > settings.MAX_SPEECH_RATE:
        return False, f"语速过快，最大值为{settings.MAX_SPEECH_RATE}"

    return True, "验证通过"


def convert_speech_rate_to_speed(speech_rate: float) -> float:
    """将阿里云speech_rate参数转换为内部speed参数

    Args:
        speech_rate: 阿里云语速参数 (-500~500)

    Returns:
        内部speed参数 (0.5~2.0)
    """
    # 阿里云speech_rate: -500(最慢) ~ 0(正常) ~ 500(最快)
    # 内部speed: 0.5(最慢) ~ 1.0(正常) ~ 2.0(最快)

    if speech_rate == 0:
        return 1.0  # 正常语速

    # 将-500~500映射到0.5~2.0
    # 使用线性映射: speech_rate = -500 -> speed = 0.5, speech_rate = 500 -> speed = 2.0
    # 公式: speed = 1.0 + (speech_rate / 500.0) * 1.0
    # 这样: -500 -> 0.0, 0 -> 1.0, 500 -> 2.0
    speed = 1.0 + (speech_rate / 500.0)

    # 确保在有效范围内
    speed = max(0.5, min(2.0, speed))

    return speed


def validate_voice_parameter(voice: str) -> Tuple[bool, str]:
    """验证音色参数

    Args:
        voice: 音色名称

    Returns:
        (is_valid, message): 验证结果和消息
    """
    if not voice or not voice.strip():
        return False, "音色参数不能为空"

    voice = voice.strip()

    if voice in settings.PRESET_VOICES:
        return True, "验证通过"

    # 如果不是预设音色，可能是自定义音色或文件路径
    return True, "验证通过"


def clean_text_for_tts(text: str) -> str:
    """清理和预处理文本，使其适合TTS合成

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    if not text:
        return ""

    # 去除首尾空白
    text = text.strip()

    # 替换多个连续空格为单个空格
    text = re.sub(r"\s+", " ", text)

    # 移除不支持的特殊字符（保留基本标点，包含中英文标点）
    # 保留中文字符、英文字母数字、空白字符、各种中英文标点符号
    text = re.sub(
        r'[^\u4e00-\u9fff\w\s.,!?;:()""' '""《》【】（）、。！？；：，\-\+\=@_]',
        "",
        text,
    )

    return text


def parse_language_code(lang_code: Optional[str]) -> str:
    """解析语言代码

    Args:
        lang_code: 语言代码（如 zh, zh-cn, en, ja等）

    Returns:
        标准化的语言代码
    """
    if not lang_code:
        return "zh"  # 默认中文

    lang_code = lang_code.lower().strip()

    # 语言代码映射
    lang_mapping = {
        "zh": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "zh-hk": "zh",
        "en": "en",
        "en-us": "en",
        "en-gb": "en",
        "ja": "jp",
        "jp": "jp",
        "ko": "kr",
        "kr": "kr",
        "yue": "yue",  # 粤语
    }

    return lang_mapping.get(lang_code, "zh")


def estimate_synthesis_time(text_length: int) -> float:
    """估算合成时间（秒）

    Args:
        text_length: 文本长度

    Returns:
        预估合成时间（秒）
    """
    # 基于经验值：平均每个字符需要约0.1秒合成时间
    base_time = text_length * 0.1

    # 考虑模型加载和其他开销
    overhead = 2.0

    return max(base_time + overhead, 3.0)  # 最少3秒
