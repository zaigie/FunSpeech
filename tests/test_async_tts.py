# -*- coding: utf-8 -*-
"""
异步TTS测试用例
测试长文本异步语音合成功能，包括回调通知机制

使用方法：
1. 完整测试（提交新任务 + 轮询结果 + 回调测试）：
   python test_async_tts.py

2. 仅轮询已有任务（在TEST_CONFIG中设置task_id）：
   修改TEST_CONFIG["task_id"] = "your_task_id_here"
   然后运行 python test_async_tts.py

3. 禁用回调测试：
   修改TEST_CONFIG["test_callback"] = False

4. 自定义回调端口：
   修改TEST_CONFIG["callback_port"] = 8888
"""

import sys
import time
import json
import requests
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

# 测试配置
TEST_CONFIG = {
    "base_url": "http://localhost:8000",
    "voice": "中文女",
    "sample_rate": 22050,
    "format": "wav",
    "enable_subtitle": True,
    "max_poll_attempts": 30,
    "poll_interval": 2.0,  # 秒
    "task_id": None,  # 如果设置了task_id，则跳过提交直接轮询
    # 回调测试配置
    "test_callback": True,  # 是否测试回调功能
    "callback_port": 8899,  # 回调服务器端口
    "callback_timeout": 60,  # 等待回调的超时时间（秒）
}

# 测试文本（较长的文本用于测试异步合成）
TEST_TEXT = """
欢迎使用FunSpeech异步语音合成服务！这是一个基于CosyVoice的高质量语音合成系统。
本系统支持多种音色选择，包括中文男声、中文女声、英文音色等。
异步合成特别适用于长文本处理，可以有效避免超时问题。
系统采用SQLite数据库存储任务状态，支持任务查询和进度跟踪。
分句功能基于CosyVoice内部逻辑，确保时间戳信息的准确性。
感谢您使用我们的语音合成服务，祝您使用愉快！
"""

# =============== 回调测试功能 ===============

# 全局变量存储回调接收到的数据
callback_received_data = []
callback_server_running = False


class CallbackHandler(BaseHTTPRequestHandler):
    """回调服务器处理器"""

    def do_POST(self):
        """处理POST请求（回调通知）"""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            # 解析JSON数据
            callback_data = json.loads(post_data.decode("utf-8"))

            # 存储回调数据
            callback_received_data.append(
                {
                    "timestamp": time.time(),
                    "data": callback_data,
                    "path": self.path,
                    "headers": dict(self.headers),
                }
            )

            print(f"\n🔔 收到回调通知:")
            print_response("回调数据", callback_data)

            # 返回成功响应
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        except Exception as e:
            print(f"❌ 处理回调请求失败: {str(e)}")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format_string, *args):
        """禁用默认的日志输出"""
        pass


def start_callback_server(port: int) -> threading.Thread:
    """启动回调服务器"""
    global callback_server_running

    def run_server():
        global callback_server_running
        try:
            server = HTTPServer(("localhost", port), CallbackHandler)
            callback_server_running = True
            print(f"🔧 回调服务器启动: http://localhost:{port}")
            server.serve_forever()
        except Exception as e:
            print(f"❌ 回调服务器启动失败: {str(e)}")
            callback_server_running = False

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # 等待服务器启动
    for _ in range(10):  # 最多等待1秒
        if callback_server_running:
            break
        time.sleep(0.1)

    return thread


def wait_for_callback(timeout: int = 60) -> Dict[str, Any]:
    """等待回调通知"""
    print_section("3. 等待回调通知")

    print(f"⏳ 等待回调通知，超时时间: {timeout} 秒")

    start_time = time.time()
    last_count = 0

    while time.time() - start_time < timeout:
        current_count = len(callback_received_data)

        if current_count > last_count:
            print(f"📨 收到 {current_count} 个回调通知")
            last_count = current_count

        # 检查是否有完成的回调（SUCCESS或FAILED）
        for callback in callback_received_data:
            data = callback["data"]
            if "error_message" in data:
                status = data.get("error_message")
                if status in ["SUCCESS", "FAILED"]:
                    print(f"✅ 收到最终状态回调: {status}")
                    return callback

        time.sleep(1)

    print(f"⏰ 等待回调超时 ({timeout} 秒)")
    if callback_received_data:
        print(f"📊 共收到 {len(callback_received_data)} 个回调通知")
        return callback_received_data[-1]  # 返回最后一个

    return None


# =============== 测试函数 ===============


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_response(response_type: str, data: Dict[str, Any]):
    """格式化打印响应数据"""
    print(f"\n【{response_type}】")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def submit_async_tts_task(
    enable_callback: bool = False, callback_url: str = None
) -> str:
    """提交异步TTS任务"""
    print_section("1. 提交异步TTS任务")

    url = f"{TEST_CONFIG['base_url']}/rest/v1/tts/async"

    # 构建请求数据（参照阿里云格式）
    request_data = {
        "payload": {
            "tts_request": {
                "voice": TEST_CONFIG["voice"],
                "sample_rate": TEST_CONFIG["sample_rate"],
                "format": TEST_CONFIG["format"],
                "text": TEST_TEXT.strip(),
                "enable_subtitle": TEST_CONFIG["enable_subtitle"],
            },
            "enable_notify": enable_callback,
        },
        "context": {"device_id": "test_device_001"},
        "header": {"appkey": "test_appkey", "token": "test_token"},
    }

    # 如果启用回调，添加回调URL
    if enable_callback and callback_url:
        request_data["payload"]["notify_url"] = callback_url

    print(f"请求URL: {url}")
    print(f"请求方法: POST")
    print(f"测试文本长度: {len(TEST_TEXT.strip())} 字符")
    if enable_callback:
        print(f"回调设置: 启用 - {callback_url}")
    else:
        print("回调设置: 禁用")

    print_response("请求数据", request_data)

    try:
        response = requests.post(url, json=request_data, timeout=30)

        print(f"\nHTTP状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print_response("成功响应", result)

            if result.get("status") == 200 and result.get("data", {}).get("task_id"):
                task_id = result["data"]["task_id"]
                print(f"\n✅ 任务提交成功!")
                print(f"任务ID: {task_id}")
                return task_id
            else:
                print(f"\n❌ 任务提交失败: {result.get('error_message', '未知错误')}")
                return None
        else:
            try:
                error_data = response.json()
                print_response("错误响应", error_data)
            except:
                print(f"响应内容: {response.text}")
            return None

    except Exception as e:
        print(f"\n❌ 请求异常: {str(e)}")
        return None


def poll_task_result(task_id: str) -> Dict[str, Any]:
    """轮询任务结果"""
    print_section("2. 轮询任务结果")

    url = f"{TEST_CONFIG['base_url']}/rest/v1/tts/async"

    params = {"appkey": "test_appkey", "token": "test_token", "task_id": task_id}

    print(f"轮询URL: {url}")
    print(f"查询参数: {params}")
    print(f"最大轮询次数: {TEST_CONFIG['max_poll_attempts']}")
    print(f"轮询间隔: {TEST_CONFIG['poll_interval']} 秒")

    for attempt in range(1, TEST_CONFIG["max_poll_attempts"] + 1):
        print(f"\n--- 第 {attempt} 次轮询 ---")

        try:
            response = requests.get(url, params=params, timeout=30)

            print(f"HTTP状态码: {response.status_code}")

            if response.status_code == 200:
                result = response.json()

                status = result.get("error_message", "UNKNOWN")
                print(f"任务状态: {status}")

                if status == "SUCCESS":
                    print_response("最终成功响应", result)
                    print(f"\n✅ 任务完成!")

                    # 检查音频地址和句子信息
                    data = result.get("data", {})
                    audio_address = data.get("audio_address")
                    sentences = data.get("sentences", [])

                    if audio_address:
                        print(f"🎵 音频地址: {audio_address}")

                    if sentences:
                        print(f"📝 句子时间戳信息 ({len(sentences)} 个句子):")
                        for i, sentence in enumerate(sentences, 1):
                            print(f"  句子{i}: \"{sentence['text']}\"")
                            print(
                                f"         时间: {sentence['begin_time']}ms - {sentence['end_time']}ms"
                            )

                    return result

                elif status == "RUNNING":
                    print_response("运行中响应", result)
                    print(
                        f"⏳ 任务正在处理中，等待 {TEST_CONFIG['poll_interval']} 秒后重试..."
                    )
                    time.sleep(TEST_CONFIG["poll_interval"])

                elif status == "FAILED":
                    print_response("失败响应", result)
                    print(f"\n❌ 任务失败!")
                    return result

                else:
                    print_response("其他状态响应", result)
                    print(f"⚠️ 未知状态: {status}")
                    time.sleep(TEST_CONFIG["poll_interval"])

            else:
                try:
                    error_data = response.json()
                    print_response("错误响应", error_data)
                except:
                    print(f"响应内容: {response.text}")

                print(f"❌ 轮询失败，HTTP状态码: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ 轮询异常: {str(e)}")

        if attempt < TEST_CONFIG["max_poll_attempts"]:
            print(f"等待 {TEST_CONFIG['poll_interval']} 秒后进行下一次轮询...")
            time.sleep(TEST_CONFIG["poll_interval"])

    print(f"\n⏰ 轮询超时，已尝试 {TEST_CONFIG['max_poll_attempts']} 次")
    return None


def test_audio_access(audio_address: str):
    """测试音频文件访问"""
    print_section("3. 测试音频文件访问")

    if not audio_address:
        print("❌ 没有可用的音频地址")
        return

    # 构建完整URL
    if audio_address.startswith("/tmp/"):
        full_url = f"{TEST_CONFIG['base_url']}{audio_address}"
    else:
        full_url = audio_address

    print(f"音频URL: {full_url}")

    try:
        response = requests.head(full_url, timeout=10)
        print(f"HTTP状态码: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print(f"Content-Length: {response.headers.get('Content-Length', 'N/A')} bytes")

        if response.status_code == 200:
            print("✅ 音频文件可正常访问")
        else:
            print(f"❌ 音频文件访问失败")

    except Exception as e:
        print(f"❌ 音频访问异常: {str(e)}")


def run_async_tts_test():
    """运行完整的异步TTS测试"""
    print_section("FunSpeech 异步TTS测试")

    print("测试配置:")
    for key, value in TEST_CONFIG.items():
        print(f"  {key}: {value}")

    # 检查是否已有task_id
    existing_task_id = TEST_CONFIG.get("task_id")
    test_callback = TEST_CONFIG.get("test_callback", False)

    # 回调测试相关变量
    callback_server_thread = None
    callback_url = None

    try:
        # 启动回调服务器（如果启用回调测试）
        if test_callback and not existing_task_id:
            callback_port = TEST_CONFIG.get("callback_port", 8899)
            callback_server_thread = start_callback_server(callback_port)

            if not callback_server_running:
                print("❌ 回调服务器启动失败，跳过回调测试")
                test_callback = False
            else:
                callback_url = f"http://localhost:{callback_port}/callback"
                print(f"✅ 回调服务器启动成功: {callback_url}")

        if existing_task_id:
            print(f"\n🔄 检测到已有task_id: {existing_task_id}")
            print("跳过任务提交，直接轮询结果...")
            task_id = existing_task_id
        else:
            # 1. 提交任务
            task_id = submit_async_tts_task(test_callback, callback_url)

            if not task_id:
                print("\n❌ 测试失败：无法提交任务")
                return False

        # 2. 处理结果获取
        if test_callback and callback_url:
            # 回调模式：等待回调通知
            callback_result = wait_for_callback(TEST_CONFIG.get("callback_timeout", 60))

            if callback_result:
                print_section("4. 回调测试结果分析")
                callback_data = callback_result["data"]

                # 检查回调数据格式
                if "data" in callback_data:
                    # 成功回调格式 (AsyncTTSResponse)
                    print("📋 回调类型: 成功响应 (AsyncTTSResponse)")
                    result = callback_data

                    # 验证notify_custom字段
                    notify_custom = callback_data.get("data", {}).get("notify_custom")
                    if notify_custom == callback_url:
                        print("✅ notify_custom字段验证通过")
                    else:
                        print(
                            f"❌ notify_custom字段不匹配: 期望 {callback_url}, 实际 {notify_custom}"
                        )

                elif "url" in callback_data:
                    # 错误回调格式 (AsyncTTSErrorResponse)
                    print("📋 回调类型: 错误响应 (AsyncTTSErrorResponse)")
                    print(
                        f"❌ 任务失败: {callback_data.get('error_message', '未知错误')}"
                    )
                    return False

                else:
                    print("❌ 未知的回调数据格式")
                    print_response("回调数据", callback_data)
                    return False

            else:
                print("❌ 未收到回调通知，切换到轮询模式")
                result = poll_task_result(task_id)
        else:
            # 轮询模式：主动查询结果
            result = poll_task_result(task_id)

        if not result:
            print("\n❌ 测试失败：无法获取结果")
            return False

        # 3. 测试音频访问
        audio_address = result.get("data", {}).get("audio_address")
        if audio_address:
            test_audio_access(audio_address)

        return analyze_test_results(
            result,
            task_id,
            test_callback,
            len(callback_received_data) if test_callback else 0,
        )

    finally:
        # 清理资源
        if callback_server_thread:
            print("\n🧹 清理回调服务器...")
            # 注意：daemon线程会在主程序退出时自动清理


def analyze_test_results(
    result: Dict[str, Any], task_id: str, test_callback: bool, callback_count: int
) -> bool:
    """分析测试结果"""
    print_section("测试总结")

    # 分析结果
    if result.get("error_message") == "SUCCESS":
        data = result.get("data", {})
        sentences = data.get("sentences", [])
        audio_address = data.get("audio_address")

        print("✅ 异步TTS测试成功完成!")
        print(f"📊 测试统计:")
        print(f"  - 输入文本长度: {len(TEST_TEXT.strip())} 字符")
        print(f"  - 生成句子数量: {len(sentences) if sentences else 0} 个")
        print(f"  - 任务ID: {task_id}")
        print(f"  - 音频地址: {audio_address or 'N/A'}")

        # 回调测试统计
        if test_callback:
            print(f"  - 回调测试: 启用")
            print(f"  - 回调通知数量: {callback_count} 个")
            notify_custom = data.get("notify_custom")
            print(f"  - notify_custom: {notify_custom or 'N/A'}")
        else:
            print(f"  - 回调测试: 禁用")

        if sentences:
            total_duration = 0
            if sentences:
                last_sentence = sentences[-1]
                total_duration = int(last_sentence.get("end_time", "0"))

            print(
                f"  - 音频总时长: {total_duration} 毫秒 ({total_duration/1000:.2f} 秒)"
            )
            print(
                f"  - 平均语速: {len(TEST_TEXT.strip())/(total_duration/1000):.1f} 字符/秒"
                if total_duration > 0
                else "  - 平均语速: N/A"
            )

        # 回调详细分析
        if test_callback and callback_count > 0:
            print(f"\n📋 回调详细分析:")
            for i, callback in enumerate(callback_received_data, 1):
                callback_data = callback["data"]
                timestamp = callback["timestamp"]
                status = callback_data.get("error_message", "UNKNOWN")
                print(f"  回调 {i}: {status} (时间戳: {timestamp:.0f})")

        return True
    else:
        print("❌ 异步TTS测试失败!")
        print(f"错误信息: {result.get('error_message', '未知错误')}")

        # 回调失败分析
        if test_callback:
            print(f"📋 回调测试统计:")
            print(f"  - 回调通知数量: {callback_count} 个")
            if callback_count > 0:
                print(
                    f"  - 最后回调状态: {callback_received_data[-1]['data'].get('error_message', 'UNKNOWN')}"
                )

        return False


if __name__ == "__main__":
    print("FunSpeech 异步TTS测试工具")
    print("=" * 60)
    print("此测试将验证异步长文本语音合成功能")
    print("包括任务提交、状态轮询、结果获取、音频访问和回调通知")
    print("=" * 60)

    try:
        success = run_async_tts_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ 测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 测试异常: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
