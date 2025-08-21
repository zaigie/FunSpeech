# -*- coding: utf-8 -*-
"""
Volume参数使用示例
展示如何使用音量参数控制TTS合成音频的音量
"""

import requests
import json

# API基础URL
BASE_URL = "http://localhost:8000"


def example_basic_volume():
    """基础音量参数使用示例"""
    print("=== 基础音量参数示例 ===")

    # 示例1: 使用默认音量(50)
    request_data = {
        "text": "这是默认音量的语音合成示例",
        "voice": "中文女",
        # volume参数可选，默认值为50
    }

    response = requests.post(f"{BASE_URL}/stream/v1/tts", json=request_data)
    print("默认音量(50):", response.json())

    # 示例2: 低音量(25)
    request_data = {
        "text": "这是低音量的语音合成示例",
        "voice": "中文女",
        "volume": 25,  # 比默认音量低
    }

    response = requests.post(f"{BASE_URL}/stream/v1/tts", json=request_data)
    print("低音量(25):", response.json())

    # 示例3: 高音量(80)
    request_data = {
        "text": "这是高音量的语音合成示例",
        "voice": "中文女",
        "volume": 80,  # 比默认音量高
    }

    response = requests.post(f"{BASE_URL}/stream/v1/tts", json=request_data)
    print("高音量(80):", response.json())


def example_volume_with_other_params():
    """音量参数与其他参数组合使用示例"""
    print("\n=== 音量参数与其他参数组合示例 ===")

    request_data = {
        "text": "这是一个完整的语音合成示例，包含音量、语速、格式等参数",
        "voice": "中文女",
        "volume": 70,  # 音量：70%
        "speech_rate": 20,  # 语速：稍快
        "format": "wav",  # 格式：WAV
        "sample_rate": 22050,  # 采样率：22050Hz
    }

    response = requests.post(f"{BASE_URL}/stream/v1/tts", json=request_data)
    result = response.json()

    print("完整参数示例:")
    print(f"  任务ID: {result.get('task_id')}")
    print(f"  音频URL: {result.get('audio_url')}")
    print(f"  状态: {result.get('message')}")


def example_openai_volume():
    """OpenAI兼容接口音量参数示例"""
    print("\n=== OpenAI兼容接口音量参数示例 ===")

    request_data = {
        "input": "This is an English TTS example with volume control",
        "voice": "英文女",
        "model": "tts-1",
        "volume": 60,  # 音量：60%
        "speed": 1.2,  # 语速：1.2倍
    }

    response = requests.post(f"{BASE_URL}/openai/v1/audio/speech", json=request_data)

    if response.status_code == 200:
        print("OpenAI兼容接口合成成功")
        print(f"  返回音频文件大小: {len(response.content)} bytes")

        # 保存音频文件
        with open("openai_volume_example.wav", "wb") as f:
            f.write(response.content)
        print("  音频已保存为: openai_volume_example.wav")
    else:
        print(f"合成失败: {response.status_code}")


def example_volume_range():
    """音量参数取值范围示例"""
    print("\n=== 音量参数取值范围示例 ===")

    volume_examples = [
        (0, "静音"),
        (25, "低音量"),
        (50, "标准音量"),
        (75, "高音量"),
        (100, "最大音量"),
    ]

    for volume, description in volume_examples:
        request_data = {
            "text": f"这是{description}的示例",
            "voice": "中文男",
            "volume": volume,
        }

        response = requests.post(f"{BASE_URL}/stream/v1/tts", json=request_data)

        if response.status_code == 200:
            result = response.json()
            print(f"  音量 {volume:3d} ({description:6s}): ✓ {result.get('task_id')}")
        else:
            print(f"  音量 {volume:3d} ({description:6s}): ✗ 失败")


if __name__ == "__main__":
    print("TTS Volume Parameter Examples")
    print("=" * 50)

    try:
        example_basic_volume()
        example_volume_with_other_params()
        example_openai_volume()
        example_volume_range()

    except requests.exceptions.ConnectionError:
        print("\n❌ 连接失败：请确保TTS服务正在运行")
        print("启动命令: python main.py")

    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")

    print("\n" + "=" * 50)
    print("示例运行完成")
