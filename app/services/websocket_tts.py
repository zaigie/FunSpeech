# -*- coding: utf-8 -*-
"""
WebSocket TTS 服务 - 阿里云流式语音合成协议实现
"""

import json
import asyncio
import logging
from typing import Optional, AsyncGenerator, Dict, Any
from enum import IntEnum

import torch
import numpy as np
from fastapi import WebSocketDisconnect

from ..core.config import settings
from ..core.executor import run_sync_generator
from ..core.security import validate_token_websocket, validate_request_appkey
from ..models.websocket_tts import (
    AliyunWSMessage,
    AliyunWSHeader,
    AliyunStartSynthesisPayload,
    AliyunRunSynthesisPayload,
    AliyunSynthesisPayload,
    AliyunSubtitle,
    AliyunTTSNamespace,
    AliyunTTSMessageName,
    AliyunTTSStatus,
)
from ..utils.common import (
    generate_task_id,
    clean_text_for_tts,
    convert_speech_rate_to_speed,
)
from ..utils.audio import validate_audio_format, validate_sample_rate
from .tts.engine import get_tts_engine

logger = logging.getLogger(__name__)


class ConnectionState(IntEnum):
    """连接状态"""

    READY = 1  # 准备就绪
    STARTED = 2  # 已开始合成，可以处理多次RunSynthesis
    COMPLETED = 3  # 已完成，只有收到StopSynthesis才会到达此状态


class AliyunWebSocketTTSService:
    """阿里云WebSocket流式TTS服务"""

    def __init__(self):
        self.tts_engine = None

    def cleanup(self):
        """清理资源"""
        try:
            if self.tts_engine:
                logger.info("正在清理WebSocket TTS引擎资源...")
                self.tts_engine.cleanup()
                logger.info("WebSocket TTS引擎资源已清理")
        except Exception as e:
            logger.warning(f"清理WebSocket TTS资源时出现异常: {e}")

        # 额外清理：导入并清理所有TTS引擎
        try:
            from .tts.engine import cleanup_all_tts_engines

            cleanup_all_tts_engines()
        except Exception as e:
            logger.warning(f"清理所有TTS引擎时出现异常: {e}")

    def _ensure_tts_engine(self):
        """确保TTS引擎已加载（懒加载）"""
        if self.tts_engine is None:
            logger.info("首次使用WebSocket TTS，正在加载TTS引擎...")
            self._initialize_engine()
        return self.tts_engine

    def _initialize_engine(self):
        """初始化TTS引擎"""
        try:
            self.tts_engine = get_tts_engine()
            logger.info("WebSocket TTS引擎加载完成")
        except Exception as e:
            logger.error(f"WebSocket TTS引擎加载失败: {e}")
            raise e

    async def _process_websocket_connection(self, websocket, task_id: str):
        """处理WebSocket连接"""
        state = ConnectionState.READY
        session_id = f"session_{task_id}"
        synthesis_params = None

        logger.debug(f"[{task_id}] 阿里云WebSocket连接开始处理")

        try:
            # 检查WebSocket头部是否有token
            if hasattr(websocket, "headers"):
                x_nls_token = websocket.headers.get("X-NLS-Token")
                if settings.APPTOKEN and not x_nls_token:
                    await self._send_task_failed(
                        websocket, task_id, "X-NLS-Token not found in ws header"
                    )
                    return

                # 验证token
                if x_nls_token:
                    result, message = validate_token_websocket(x_nls_token, task_id)
                    if not result:
                        await self._send_task_failed(websocket, task_id, message)
                        return

            while True:
                # 等待接收消息
                message = await websocket.receive_text()

                try:
                    # 解析消息
                    data = json.loads(message)
                    logger.debug(f"[{task_id}] 收到消息: {data}")

                    header = data.get("header", {})
                    message_name = header.get("name", "")
                    message_task_id = header.get("task_id", "")
                    namespace = header.get("namespace", "")

                    # 验证namespace
                    if namespace != AliyunTTSNamespace.FLOWING_SPEECH_SYNTHESIZER:
                        await self._send_task_failed(
                            websocket, task_id, "Invalid namespace"
                        )
                        continue

                    # 处理不同的消息类型
                    if message_name == AliyunTTSMessageName.START_SYNTHESIS:
                        if state == ConnectionState.READY:
                            # 验证和保存合成参数
                            synthesis_params = self._parse_start_synthesis(
                                data, task_id
                            )
                            if synthesis_params:
                                task_id = message_task_id or task_id
                                # 发送开始响应
                                await self._send_synthesis_started(
                                    websocket, task_id, session_id
                                )
                                state = ConnectionState.STARTED
                            else:
                                await self._send_task_failed(
                                    websocket,
                                    task_id,
                                    "Invalid StartSynthesis parameters",
                                )
                        else:
                            await self._send_task_failed(
                                websocket, task_id, "Connection already started"
                            )

                    elif message_name == AliyunTTSMessageName.RUN_SYNTHESIS:
                        if state == ConnectionState.STARTED:
                            if message_task_id != task_id:
                                await self._send_task_failed(
                                    websocket, task_id, "Task ID not match"
                                )
                                continue

                            # 执行流式合成 - 支持多次调用
                            text = data.get("payload", {}).get("text", "")
                            if text and synthesis_params:
                                await self._run_synthesis(
                                    websocket,
                                    task_id,
                                    session_id,
                                    text,
                                    synthesis_params,
                                )
                                # 注意：state保持STARTED，允许后续继续发送RunSynthesis
                            else:
                                await self._send_task_failed(
                                    websocket, task_id, "Missing text in RunSynthesis"
                                )
                        else:
                            await self._send_task_failed(
                                websocket, task_id, "Connection not started"
                            )

                    elif message_name == AliyunTTSMessageName.STOP_SYNTHESIS:
                        if state == ConnectionState.STARTED:
                            if message_task_id != task_id:
                                await self._send_task_failed(
                                    websocket, task_id, "Task ID not match"
                                )
                                continue

                            # 完成合成
                            await self._send_synthesis_completed(
                                websocket, task_id, session_id
                            )
                            state = ConnectionState.COMPLETED
                            logger.debug(f"[{task_id}] 阿里云WebSocket合成完成")
                            break
                        else:
                            await self._send_task_failed(
                                websocket, task_id, "Connection not started"
                            )
                    else:
                        await self._send_task_failed(
                            websocket, task_id, f"Invalid message name: {message_name}"
                        )

                except json.JSONDecodeError as e:
                    logger.error(f"[{task_id}] JSON解析错误: {e}")
                    await self._send_task_failed(
                        websocket, task_id, f"Message Not Json: {message}"
                    )
                except WebSocketDisconnect:
                    # 客户端断开，向外层抛出
                    logger.debug(f"[{task_id}] 处理消息时检测到客户端断开")
                    raise
                except Exception as e:
                    logger.error(f"[{task_id}] 处理消息异常: {e}")
                    await self._send_task_failed(websocket, task_id, str(e))
                    break

        except WebSocketDisconnect:
            logger.warning(f"[{task_id}] 客户端主动断开WebSocket连接")
        except Exception as e:
            # 检查是否是WebSocket连接相关的异常
            error_msg = str(e)
            if "WebSocket is not connected" in error_msg or "Need to call \"accept\" first" in error_msg:
                logger.warning(f"[{task_id}] WebSocket连接已断开: {e}")
            else:
                logger.error(f"[{task_id}] WebSocket连接处理异常: {e}")
                try:
                    await self._send_task_failed(websocket, task_id, str(e))
                except:
                    pass

    def _parse_start_synthesis(self, data: dict, task_id: str) -> Optional[dict]:
        """解析StartSynthesis消息"""
        try:
            payload = data.get("payload", {})

            # 构建合成参数
            params = {
                "voice": payload.get("voice", "中文女"),
                "format": payload.get("format", "PCM"),
                "sample_rate": payload.get("sample_rate", 22050),
                "volume": payload.get("volume", 50),
                "speech_rate": payload.get("speech_rate", 0),
                "pitch_rate": payload.get("pitch_rate", 0),
                "enable_subtitle": payload.get("enable_subtitle", False),
            }

            # 验证参数
            if not validate_audio_format(params["format"].lower()):
                return None
            if not validate_sample_rate(params["sample_rate"]):
                return None

            logger.info(f"[{task_id}] StartSynthesis参数解析成功: {params}")
            return params

        except Exception as e:
            logger.error(f"[{task_id}] 解析StartSynthesis失败: {e}")
            return None

    async def _run_synthesis(
        self, websocket, task_id: str, session_id: str, text: str, params: dict
    ):
        """执行流式合成"""
        logger.debug(f"[{task_id}] 开始流式合成文本: '{text}'")

        try:
            # 发送句子开始
            await self._send_sentence_begin(websocket, task_id, session_id)

            # 清理文本
            clean_text = clean_text_for_tts(text)
            speed = convert_speech_rate_to_speed(params["speech_rate"])

            # 映射本地音色到阿里云音色名称
            local_voice = self._map_aliyun_voice_to_local(params["voice"])

            # 生成音频
            audio_sent = False
            async for audio_chunk in self._synthesize_streaming_audio(
                clean_text,
                local_voice,
                speed,
                params["format"],
                params["sample_rate"],
                params["volume"],
                task_id,
                websocket,  # 传入websocket用于检测连接状态
            ):
                if audio_chunk and len(audio_chunk) > 0:
                    # 检查WebSocket连接状态
                    if websocket.client_state.name != "CONNECTED":
                        logger.warning(
                            f"[{task_id}] 检测到客户端已断开，停止合成"
                        )
                        return

                    # 发送音频数据（二进制）
                    await websocket.send_bytes(audio_chunk)
                    audio_sent = True

                    # 发送句子合成进度（可选）
                    await self._send_sentence_synthesis(
                        websocket, task_id, session_id, text
                    )

                    # 小延迟模拟流式效果
                    await asyncio.sleep(0.05)

            if not audio_sent:
                logger.warning(f"[{task_id}] 没有生成任何音频数据")

            # 发送句子结束
            await self._send_sentence_end(websocket, task_id, session_id, text)

        except WebSocketDisconnect:
            logger.warning(f"[{task_id}] 客户端在合成过程中断开连接")
            raise
        except Exception as e:
            # 检查是否是因为WebSocket断开导致的异常
            if "WebSocket is not connected" in str(e) or "Need to call \"accept\" first" in str(e):
                logger.warning(f"[{task_id}] 客户端已断开连接: {e}")
                raise WebSocketDisconnect()
            else:
                logger.error(f"[{task_id}] 流式合成失败: {e}")
                await self._send_task_failed(
                    websocket, task_id, f"Synthesis failed: {str(e)}"
                )

    def _map_aliyun_voice_to_local(self, aliyun_voice: str) -> str:
        """直接返回音色名称，不进行映射"""
        return aliyun_voice

    async def _synthesize_streaming_audio(
        self,
        text: str,
        voice: str,
        speed: float,
        format: str,
        sample_rate: int,
        volume: int,
        task_id: str,
        websocket,  # 添加websocket参数用于检测连接状态
    ) -> AsyncGenerator[Optional[bytes], None]:
        """生成流式音频数据"""
        try:
            tts_engine = self._ensure_tts_engine()

            # 检查是否为零样本克隆音色
            if (
                tts_engine._voice_manager
                and tts_engine._voice_manager.is_voice_available(voice)
            ):
                if voice in tts_engine._voice_manager.list_clone_voices():
                    # 使用CosyVoice2流式合成（零样本克隆音色）
                    async for chunk in self._stream_clone_voice(
                        text, voice, speed, format, task_id, websocket
                    ):
                        yield chunk
                    return

            # 使用CosyVoice1流式合成（预设音色）
            if tts_engine.cosyvoice_sft:
                async for chunk in self._stream_preset_voice(
                    text, voice, speed, format, task_id, websocket
                ):
                    yield chunk
            else:
                raise Exception("预设音色模型未加载")

        except WebSocketDisconnect:
            logger.warning(f"[{task_id}] 客户端断开，停止音频生成")
            raise
        except Exception as e:
            logger.error(f"[{task_id}] 流式合成失败: {e}")
            raise e

    async def _stream_preset_voice(
        self, text: str, voice: str, speed: float, format: str, task_id: str, websocket
    ) -> AsyncGenerator[bytes, None]:
        """使用CosyVoice1进行流式合成（预设音色）"""
        logger.debug(f"[{task_id}] 使用CosyVoice1流式合成预设音色: {voice}")

        tts_engine = self._ensure_tts_engine()

        # 使用线程池执行流式推理，避免阻塞事件循环
        async for audio_data in run_sync_generator(
            tts_engine.cosyvoice_sft.inference_sft,
            text, voice, stream=True, speed=speed
        ):
            # 检查连接状态，如果断开则立即停止
            if websocket.client_state.name != "CONNECTED":
                logger.warning(f"[{task_id}] 客户端已断开，停止预设音色合成")
                return

            # 将tensor转换为numpy数组
            audio_array = audio_data["tts_speech"].numpy()

            # 根据格式转换音频数据
            if format.upper() == "PCM":
                # PCM格式：直接发送16位PCM数据
                pcm_bytes = self._convert_audio_to_pcm(
                    audio_array, tts_engine.cosyvoice_sft.sample_rate
                )
                yield pcm_bytes
            else:
                # WAV格式：转换为WAV字节流
                wav_bytes = self._convert_audio_to_wav(
                    audio_array, tts_engine.cosyvoice_sft.sample_rate
                )
                yield wav_bytes

            # 添加小延迟以模拟真实流式效果
            await asyncio.sleep(0.01)

    async def _stream_clone_voice(
        self, text: str, voice: str, speed: float, format: str, task_id: str, websocket
    ) -> AsyncGenerator[bytes, None]:
        """使用CosyVoice2进行流式合成（零样本克隆音色）"""
        logger.debug(f"[{task_id}] 使用CosyVoice2流式合成零样本克隆音色: {voice}")

        tts_engine = self._ensure_tts_engine()

        # 使用线程池执行流式推理，避免阻塞事件循环
        async for audio_data in run_sync_generator(
            tts_engine.cosyvoice_clone.inference_zero_shot,
            text,
            "",  # prompt_text - 使用空字符串，由于我们使用保存的音色
            None,  # prompt_speech_16k - 不需要，因为使用保存的音色
            zero_shot_spk_id=voice,  # 使用保存的音色ID
            stream=True,  # 关键：启用流式模式
            speed=speed,
        ):
            # 检查连接状态，如果断开则立即停止
            if websocket.client_state.name != "CONNECTED":
                logger.warning(f"[{task_id}] 客户端已断开，停止克隆音色合成")
                return

            # 将tensor转换为numpy数组
            audio_array = audio_data["tts_speech"].numpy()

            # 根据格式转换音频数据
            if format.upper() == "PCM":
                # PCM格式：直接发送16位PCM数据
                pcm_bytes = self._convert_audio_to_pcm(
                    audio_array, tts_engine.cosyvoice_clone.sample_rate
                )
                yield pcm_bytes
            else:
                # WAV格式：转换为WAV字节流
                wav_bytes = self._convert_audio_to_wav(
                    audio_array, tts_engine.cosyvoice_clone.sample_rate
                )
                yield wav_bytes

            # 添加小延迟以模拟真实流式效果
            await asyncio.sleep(0.01)

    def _convert_audio_to_pcm(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """将音频数组转换为PCM字节流"""
        try:
            if audio_array is None or audio_array.size == 0:
                return b""

            # 确保音频数据在正确的范围内
            audio_array = np.clip(audio_array, -1.0, 1.0)

            # 转换为16位PCM
            pcm_data = (audio_array * 32767).astype(np.int16)

            return pcm_data.tobytes()

        except Exception as e:
            logger.error(f"PCM转换失败: {e}")
            return b""

    def _convert_audio_to_wav(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """将音频数组转换为WAV字节流"""
        import soundfile as sf
        import io

        try:
            if audio_array is None or audio_array.size == 0:
                return b""

            # 创建内存中的字节流
            buffer = io.BytesIO()

            # 将音频数组写入内存缓冲区（WAV格式）
            sf.write(buffer, audio_array.T, sample_rate, format="WAV")

            # 获取字节数据
            buffer.seek(0)
            return buffer.read()

        except Exception as e:
            logger.error(f"WAV转换失败: {e}")
            return b""

    async def _send_synthesis_started(self, websocket, task_id: str, session_id: str):
        """发送SynthesisStarted响应"""
        response = {
            "header": {
                "message_id": AliyunWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunTTSNamespace.FLOWING_SPEECH_SYNTHESIZER,
                "name": AliyunTTSMessageName.SYNTHESIS_STARTED,
                "status": AliyunTTSStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "session_id": session_id,
                "index": 1,
            },
        }
        await websocket.send_text(json.dumps(response))
        logger.debug(f"[{task_id}] 发送SynthesisStarted")

    async def _send_sentence_begin(self, websocket, task_id: str, session_id: str):
        """发送SentenceBegin响应"""
        response = {
            "header": {
                "message_id": AliyunWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunTTSNamespace.FLOWING_SPEECH_SYNTHESIZER,
                "name": AliyunTTSMessageName.SENTENCE_BEGIN,
                "status": AliyunTTSStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "session_id": session_id,
                "index": 1,
            },
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))
        logger.debug(f"[{task_id}] 发送SentenceBegin")

    async def _send_sentence_synthesis(
        self, websocket, task_id: str, session_id: str, text: str
    ):
        """发送SentenceSynthesis响应"""
        response = {
            "header": {
                "message_id": AliyunWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunTTSNamespace.FLOWING_SPEECH_SYNTHESIZER,
                "name": AliyunTTSMessageName.SENTENCE_SYNTHESIS,
                "status": AliyunTTSStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "subtitles": [
                    {
                        "text": text,
                        "begin_time": 0,
                        "end_time": len(text) * 200,  # 估算时长
                        "begin_index": 0,
                        "end_index": len(text),
                        "sentence": True,
                        "phoneme_list": [],
                    }
                ]
            },
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))
        logger.debug(f"[{task_id}] 发送SentenceSynthesis")

    async def _send_sentence_end(
        self, websocket, task_id: str, session_id: str, text: str
    ):
        """发送SentenceEnd响应"""
        response = {
            "header": {
                "message_id": AliyunWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunTTSNamespace.FLOWING_SPEECH_SYNTHESIZER,
                "name": AliyunTTSMessageName.SENTENCE_END,
                "status": AliyunTTSStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "subtitles": [
                    {
                        "text": text,
                        "begin_time": 0,
                        "end_time": len(text) * 200,  # 估算时长
                        "begin_index": 0,
                        "end_index": len(text),
                        "sentence": True,
                        "phoneme_list": [],
                    }
                ]
            },
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))
        logger.debug(f"[{task_id}] 发送SentenceEnd")

    async def _send_synthesis_completed(self, websocket, task_id: str, session_id: str):
        """发送SynthesisCompleted响应"""
        response = {
            "header": {
                "message_id": AliyunWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunTTSNamespace.FLOWING_SPEECH_SYNTHESIZER,
                "name": AliyunTTSMessageName.SYNTHESIS_COMPLETED,
                "status": AliyunTTSStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "session_id": session_id,
                "index": 1,
            },
        }
        await websocket.send_text(json.dumps(response))
        logger.debug(f"[{task_id}] 发送SynthesisCompleted")

    async def _send_task_failed(self, websocket, task_id: str, reason: str):
        """发送TaskFailed响应"""
        response = {
            "header": {
                "namespace": AliyunTTSNamespace.DEFAULT,
                "name": AliyunTTSMessageName.TASK_FAILED,
                "status": AliyunTTSStatus.TASK_FAILED,
                "message_id": AliyunWSHeader.generate_message_id(),
                "task_id": task_id,
                "status_text": reason,
            }
        }
        try:
            await websocket.send_text(json.dumps(response))
            logger.error(f"[{task_id}] 发送TaskFailed: {reason}")
        except:
            pass


# 全局服务实例
_aliyun_websocket_tts_service = None


def get_aliyun_websocket_tts_service() -> AliyunWebSocketTTSService:
    """获取阿里云WebSocket TTS服务实例"""
    global _aliyun_websocket_tts_service
    if _aliyun_websocket_tts_service is None:
        _aliyun_websocket_tts_service = AliyunWebSocketTTSService()
    return _aliyun_websocket_tts_service
