# -*- coding: utf-8 -*-
"""
统一音频处理工具
整合ASR和TTS的音频处理功能
"""

import os
import tempfile
import requests
import librosa
import soundfile as sf
import torchaudio
import torch
import numpy as np
import subprocess
from typing import Tuple, Optional, Union
from io import BytesIO
from pathlib import Path

from ..core.config import settings
from ..core.exceptions import (
    InvalidParameterException,
    InvalidMessageException,
    AudioProcessingException,
)


def validate_audio_format(format_str: Optional[str]) -> bool:
    """验证音频格式是否支持"""
    if not format_str:
        return True  # 如果未指定格式，允许通过

    return format_str.lower() in [
        fmt.lower() for fmt in settings.SUPPORTED_AUDIO_FORMATS
    ]


def validate_sample_rate(sample_rate: Optional[int]) -> bool:
    """验证采样率是否支持"""
    if not sample_rate:
        return True  # 如果未指定采样率，允许通过

    return sample_rate in settings.SUPPORTED_SAMPLE_RATES


def download_audio_from_url(url: str, max_size: int = None) -> bytes:
    """从URL下载音频文件

    Args:
        url: 音频文件URL
        max_size: 最大文件大小限制

    Returns:
        音频文件的二进制数据

    Raises:
        InvalidParameterException: URL无效或下载失败
        InvalidMessageException: 文件太大
    """
    if not url:
        raise InvalidParameterException("URL不能为空")

    max_file_size = max_size or settings.MAX_AUDIO_SIZE

    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # 检查Content-Length头
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > max_file_size:
            max_size_mb = max_file_size // 1024 // 1024
            raise InvalidMessageException(f"音频文件太大，最大支持{max_size_mb}MB")

        # 分块下载并检查大小
        audio_data = BytesIO()
        downloaded_size = 0

        for chunk in response.iter_content(chunk_size=8192):
            downloaded_size += len(chunk)
            if downloaded_size > max_file_size:
                max_size_mb = max_file_size // 1024 // 1024
                raise InvalidMessageException(f"音频文件太大，最大支持{max_size_mb}MB")
            audio_data.write(chunk)

        return audio_data.getvalue()

    except requests.RequestException as e:
        raise InvalidParameterException(f"下载音频文件失败: {str(e)}")


def save_audio_to_temp_file(audio_data: bytes, suffix: str = ".wav") -> str:
    """保存音频数据到临时文件

    Args:
        audio_data: 音频二进制数据
        suffix: 文件后缀

    Returns:
        临时文件路径

    Raises:
        AudioProcessingException: 保存失败
    """
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=settings.TEMP_DIR
        ) as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    except Exception as e:
        raise AudioProcessingException(f"保存音频文件失败: {str(e)}")


def cleanup_temp_file(file_path: str) -> None:
    """清理临时文件

    Args:
        file_path: 文件路径
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        # 静默忽略清理错误
        pass


def load_audio_file(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """加载音频文件并转换为指定采样率

    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率

    Returns:
        (audio_data, sample_rate): 音频数据和采样率

    Raises:
        AudioProcessingException: 加载失败
    """
    try:
        # 使用librosa加载音频
        audio_data, sr = librosa.load(audio_path, sr=target_sr)
        return audio_data, sr
    except Exception as e:
        raise AudioProcessingException(f"加载音频文件失败: {str(e)}")


def get_audio_duration(audio_path: str) -> float:
    """获取音频文件时长

    Args:
        audio_path: 音频文件路径

    Returns:
        音频时长（秒）

    Raises:
        AudioProcessingException: 获取时长失败
    """
    try:
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        raise AudioProcessingException(f"获取音频时长失败: {str(e)}")


def validate_reference_audio(audio_path: str) -> Tuple[bool, str]:
    """验证参考音频文件

    Args:
        audio_path: 音频文件路径

    Returns:
        (is_valid, message): 验证结果和消息
    """
    if not audio_path or not os.path.exists(audio_path):
        return False, "音频文件不存在"

    try:
        # 检查文件大小
        file_size = os.path.getsize(audio_path)
        if file_size > settings.MAX_AUDIO_SIZE:
            max_size_mb = settings.MAX_AUDIO_SIZE // 1024 // 1024
            return False, f"音频文件太大，最大支持{max_size_mb}MB"

        # 检查音频时长
        duration = get_audio_duration(audio_path)
        if duration < settings.MIN_REFERENCE_AUDIO_DURATION:
            return (
                False,
                f"参考音频太短，最小时长{settings.MIN_REFERENCE_AUDIO_DURATION}秒",
            )

        if duration > settings.MAX_REFERENCE_AUDIO_DURATION:
            return (
                False,
                f"参考音频太长，最大时长{settings.MAX_REFERENCE_AUDIO_DURATION}秒",
            )

        return True, "验证通过"

    except Exception as e:
        return False, f"音频文件验证失败: {str(e)}"


def save_audio_array(
    audio_array: np.ndarray,
    output_path: str,
    sample_rate: int = 22050,
    format: str = "wav",
) -> str:
    """保存音频数组到文件

    Args:
        audio_array: 音频数据数组
        output_path: 输出文件路径
        sample_rate: 采样率
        format: 音频格式

    Returns:
        保存的文件路径

    Raises:
        AudioProcessingException: 保存失败
    """
    try:
        # 确保音频数据是float32格式
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # 确保音频数据在正确的范围内
        if np.max(np.abs(audio_array)) > 1.0:
            audio_array = audio_array / np.max(np.abs(audio_array))

        # 确保是2D张量 (channels, samples)
        if audio_array.ndim == 1:
            audio_array = audio_array[np.newaxis, :]  # 添加通道维度
        elif audio_array.ndim > 2:
            audio_array = audio_array.squeeze()
            if audio_array.ndim == 1:
                audio_array = audio_array[np.newaxis, :]

        # 转换为torch张量
        audio_tensor = torch.from_numpy(audio_array)

        # 使用torchaudio保存，它对格式兼容性更好
        torchaudio.save(output_path, audio_tensor, sample_rate)
        return output_path

    except Exception as e:
        raise AudioProcessingException(f"保存音频文件失败: {str(e)}")


def convert_audio_to_wav(input_path: str, output_path: str = None) -> str:
    """转换音频文件为WAV格式

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（可选）

    Returns:
        转换后的文件路径

    Raises:
        AudioProcessingException: 转换失败
    """
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".wav"

    try:
        # 使用librosa加载并保存为WAV
        audio_data, sr = librosa.load(input_path, sr=None)
        sf.write(output_path, audio_data, sr, format="wav")
        return output_path

    except Exception as e:
        # 尝试使用ffmpeg转换
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    input_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    output_path,
                    "-y",
                ],
                check=True,
                capture_output=True,
            )
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise AudioProcessingException(f"音频格式转换失败: {str(e)}")


def generate_temp_audio_path(prefix: str = "audio", suffix: str = ".wav") -> str:
    """生成临时音频文件路径

    Args:
        prefix: 文件名前缀
        suffix: 文件后缀

    Returns:
        临时文件路径
    """
    import time

    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}_{os.getpid()}{suffix}"
    return os.path.join(settings.TEMP_DIR, filename)


def cleanup_temp_audio_file(file_path: str) -> None:
    """清理临时音频文件（TTS专用）

    Args:
        file_path: 文件路径
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        # 静默忽略清理错误
        pass
