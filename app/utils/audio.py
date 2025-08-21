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
import logging
from typing import Tuple, Optional, Union
from io import BytesIO
from pathlib import Path

from ..core.config import settings
from ..core.exceptions import (
    InvalidParameterException,
    InvalidMessageException,
    DefaultServerErrorException,
)

logger = logging.getLogger(__name__)


def validate_audio_format(format_str: Optional[str]) -> bool:
    """验证音频格式是否支持"""
    if not format_str:
        return True  # 如果未指定格式，允许通过

    # 统一转换为小写进行比较
    format_lower = format_str.lower()
    supported_formats = [fmt.lower() for fmt in settings.SUPPORTED_AUDIO_FORMATS]

    return format_lower in supported_formats


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
        raise DefaultServerErrorException(f"保存音频文件失败: {str(e)}")


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
        raise DefaultServerErrorException(f"加载音频文件失败: {str(e)}")


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
        # Load audio and get duration
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        raise DefaultServerErrorException(f"获取音频时长失败: {str(e)}")


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


def resample_audio_array(
    audio_array: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """重采样音频数组

    Args:
        audio_array: 原始音频数据
        original_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        重采样后的音频数据
    """
    if original_sr == target_sr:
        return audio_array

    try:
        # 确保是1D数组用于librosa重采样
        if audio_array.ndim > 1:
            # 如果是多声道，取第一个声道
            if audio_array.shape[0] > audio_array.shape[1]:
                audio_1d = audio_array[0, :]
            else:
                audio_1d = (
                    audio_array[:, 0]
                    if audio_array.shape[1] > 1
                    else audio_array.flatten()
                )
        else:
            audio_1d = audio_array

        # 使用librosa进行重采样
        resampled = librosa.resample(audio_1d, orig_sr=original_sr, target_sr=target_sr)

        logger.info(f"音频重采样: {original_sr}Hz -> {target_sr}Hz")
        return resampled

    except Exception as e:
        logger.warning(f"音频重采样失败: {str(e)}，使用原始音频")
        return audio_array


def adjust_audio_volume(audio_array: np.ndarray, volume: int) -> np.ndarray:
    """调节音频音量

    Args:
        audio_array: 音频数据数组
        volume: 音量值，范围0~100，50为原始音量

    Returns:
        调节后的音频数据
    """
    if int(volume) == 50:
        return audio_array

    if volume < 0 or volume > 100:
        logger.warning(f"音量值{volume}超出范围[0,100]，使用默认值50")
        volume = 50

    # 将音量值转换为倍数 (0-100 -> 0-2.0)
    volume_factor = volume / 50.0

    # 应用音量调节
    adjusted_audio = audio_array * volume_factor

    # 防止削波，如果音量过大导致超过范围，进行归一化
    max_val = np.max(np.abs(adjusted_audio))
    if max_val > 1.0:
        adjusted_audio = adjusted_audio / max_val
        logger.info(f"音量调节后进行归一化，最大值: {max_val:.3f}")

    logger.info(f"音频音量已调节: {volume}/100 (倍数: {volume_factor:.2f})")
    return adjusted_audio


def save_audio_array(
    audio_array: np.ndarray,
    output_path: str,
    sample_rate: int = 22050,
    format: str = "wav",
    original_sr: int = None,
    volume: int = 50,
) -> str:
    """保存音频数组到文件

    Args:
        audio_array: 音频数据数组
        output_path: 输出文件路径
        sample_rate: 目标采样率
        format: 音频格式
        original_sr: 原始采样率（用于重采样）
        volume: 音量值，范围0~100，默认50

    Returns:
        保存的文件路径

    Raises:
        AudioProcessingException: 保存失败
    """
    try:
        # 如果指定了原始采样率且与目标采样率不同，进行重采样
        if original_sr and original_sr != sample_rate:
            audio_array = resample_audio_array(audio_array, original_sr, sample_rate)

        # 调节音频音量
        audio_array = adjust_audio_volume(audio_array, volume)

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

        # 根据格式选择保存方法
        if format.lower() == "wav":
            # 使用torchaudio保存WAV格式
            audio_tensor = torch.from_numpy(audio_array)
            torchaudio.save(output_path, audio_tensor, sample_rate)
        else:
            # 使用soundfile保存其他格式
            # 确保音频数据是单声道
            if audio_array.shape[0] > 1:
                audio_array = np.mean(audio_array, axis=0)

            sf.write(output_path, audio_array.T, sample_rate, format=format.upper())

        return output_path

    except Exception as e:
        raise DefaultServerErrorException(f"保存音频文件失败: {str(e)}")


def convert_audio_to_wav(
    input_path: str, output_path: str = None, target_sr: int = 16000
) -> str:
    """转换音频文件为WAV格式

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（可选）
        target_sr: 目标采样率，默认16000Hz

    Returns:
        转换后的文件路径

    Raises:
        AudioProcessingException: 转换失败
    """
    if not output_path:
        output_path = input_path.rsplit(".", 1)[0] + ".wav"

    try:
        # 使用librosa加载并重采样
        audio_data, sr = librosa.load(input_path, sr=target_sr)
        sf.write(output_path, audio_data, target_sr, format="WAV")
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
                    str(target_sr),
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
            raise DefaultServerErrorException(f"音频格式转换失败: {str(e)}")


def normalize_audio_for_asr(audio_path: str, target_sr: int = 16000) -> str:
    """将音频文件标准化为ASR模型所需的格式

    Args:
        audio_path: 输入音频文件路径
        target_sr: 目标采样率，默认16000Hz

    Returns:
        标准化后的WAV文件路径

    Raises:
        AudioProcessingException: 标准化失败
    """
    try:
        # 检查文件扩展名
        file_ext = os.path.splitext(audio_path)[1].lower()

        # 如果已经是WAV格式且采样率正确，直接返回
        if file_ext == ".wav":
            # 检查采样率
            audio_data, sr = librosa.load(audio_path, sr=None)
            if sr == target_sr:
                return audio_path

        # 转换为标准WAV格式
        normalized_path = convert_audio_to_wav(audio_path, target_sr=target_sr)
        logger.info(f"音频文件已标准化: {audio_path} -> {normalized_path}")
        return normalized_path

    except Exception as e:
        raise DefaultServerErrorException(f"音频标准化失败: {str(e)}")


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


def get_audio_file_suffix(
    audio_address: Optional[str] = None, format_param: Optional[str] = None
) -> str:
    """根据音频地址和format参数生成文件后缀

    Args:
        audio_address: 音频文件地址（可选）
        format_param: format参数（可选）

    Returns:
        文件后缀（包含点号）
    """
    if audio_address and format_param:
        # 使用audio_address时，format参数生效
        return f".{format_param.lower()}"
    elif audio_address:
        # 使用audio_address但未指定format，默认为pcm
        return ".pcm"
    else:
        # 使用二进制音频流时，format参数不生效，默认为wav
        return ".wav"


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
