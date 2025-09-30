# -*- coding: utf-8 -*-
"""
WebSocket ASR 服务 - 阿里云实时语音识别协议实现

本服务实现了阿里云WebSocket实时语音识别协议，使用FunASR开源模型替代阿里云官方API。

【重要限制】
由于FunASR开源模型的能力限制，以下字段未实现:
- words字段: 词级别的时间戳信息
- confidence字段: 识别结果的置信度

【VAD与句子边界检测机制】
1. VAD机制: 完全依赖FunASR模型内置的VAD能力
2. SentenceBegin触发: 首次收到非空识别结果时
3. SentenceEnd触发:
   - 检测到句末标点符号(。！？.!?…)
   - 连续N次收到空识别结果(基于max_sentence_silence参数)
   - 收到StopTranscription指令
4. 中间结果去重: 自动去除FunASR流式识别中的重复文本
5. 缓存刷新: 句子结束时强制flush模型缓存，确保获取完整内容

【标点恢复机制】
1. 流式识别中间结果：
   - ASR_ENABLE_REALTIME_PUNC=True时，使用实时标点模型添加句内标点（逗号等）
   - ASR_ENABLE_REALTIME_PUNC=False时（默认），中间结果不添加标点
2. 句子结束时：始终使用离线标点模型对无标点文本添加完整标点（包括句末标点）
3. 双轨处理：同时维护带标点版本（展示用）和无标点版本（最终标点恢复用）
"""

import json
import logging
import numpy as np
import soundfile as sf
import io
from typing import Optional, Dict
from enum import IntEnum

from fastapi import WebSocketDisconnect

from ..core.config import settings
from ..core.security import validate_token_websocket
from ..utils.text_processing import apply_itn_to_text
from ..models.websocket_asr import (
    AliyunASRWSHeader,
    AliyunASRNamespace,
    AliyunASRMessageName,
    AliyunASRStatus,
)

logger = logging.getLogger(__name__)


class ConnectionState(IntEnum):
    """连接状态"""

    READY = 1
    STARTED = 2
    COMPLETED = 3


class AliyunWebSocketASRService:
    """阿里云WebSocket实时ASR服务"""

    def __init__(self):
        self.asr_engine = None

    def cleanup(self):
        """清理资源"""
        try:
            if self.asr_engine:
                logger.info("WebSocket ASR引擎资源已清理")
        except Exception as e:
            logger.warning(f"清理WebSocket ASR资源异常: {e}")

    def _ensure_asr_engine(self):
        """确保ASR引擎已加载"""
        if self.asr_engine is None:
            self._initialize_engine()
        return self.asr_engine

    def _initialize_engine(self):
        """初始化ASR引擎"""
        try:
            from .asr.manager import get_model_manager

            model_manager = get_model_manager()
            self.asr_engine = model_manager.get_asr_engine()

            if not self.asr_engine.supports_realtime:
                raise Exception("当前ASR引擎不支持实时识别")

            logger.info("WebSocket ASR引擎加载完成")
        except Exception as e:
            logger.error(f"WebSocket ASR引擎加载失败: {e}")
            raise e

    async def _process_websocket_connection(self, websocket, task_id: str):
        """处理WebSocket连接"""
        state = ConnectionState.READY
        session_id = f"session_{task_id}"
        transcription_params = None
        audio_cache = {}
        punc_cache = {}
        sentence_index = 0
        audio_time = 0
        sentence_active = False
        sentence_start_time = 0
        last_sentence_text = ""
        sentence_texts = []
        sentence_texts_raw = []
        empty_result_count = 0

        logger.info(f"[{task_id}] WebSocket ASR连接开始")

        try:
            if hasattr(websocket, "headers"):
                x_nls_token = websocket.headers.get("X-NLS-Token")
                if settings.APPTOKEN and not x_nls_token:
                    await self._send_task_failed(
                        websocket, task_id, "X-NLS-Token not found in ws header"
                    )
                    return

                if x_nls_token:
                    result, message = validate_token_websocket(x_nls_token, task_id)
                    if not result:
                        await self._send_task_failed(websocket, task_id, message)
                        return

            while True:
                message = await websocket.receive()

                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        logger.debug(
                            f"[{task_id}] 收到消息: {data.get('header', {}).get('name', '')}"
                        )

                        header = data.get("header", {})
                        message_name = header.get("name", "")
                        message_task_id = header.get("task_id", "")
                        namespace = header.get("namespace", "")

                        if namespace != AliyunASRNamespace.SPEECH_TRANSCRIBER:
                            await self._send_task_failed(
                                websocket, task_id, "Invalid namespace"
                            )
                            continue

                        if message_name == AliyunASRMessageName.START_TRANSCRIPTION:
                            if state == ConnectionState.READY:
                                transcription_params = self._parse_start_transcription(
                                    data, task_id
                                )
                                if transcription_params:
                                    task_id = message_task_id or task_id
                                    await self._send_transcription_started(
                                        websocket, task_id, session_id
                                    )
                                    state = ConnectionState.STARTED
                                    sentence_index = 0
                                    audio_time = 0
                                    sentence_active = False
                                    sentence_start_time = 0
                                    last_sentence_text = ""
                                    sentence_texts = []
                                    sentence_texts_raw = []
                                    empty_result_count = 0
                                else:
                                    await self._send_task_failed(
                                        websocket,
                                        task_id,
                                        "Invalid StartTranscription parameters",
                                    )
                            else:
                                await self._send_task_failed(
                                    websocket, task_id, "Connection already started"
                                )

                        elif message_name == AliyunASRMessageName.STOP_TRANSCRIPTION:
                            if state == ConnectionState.STARTED:
                                if message_task_id != task_id:
                                    await self._send_task_failed(
                                        websocket, task_id, "Task ID not match"
                                    )
                                    continue

                                # 如果有未完成的句子，直接结束
                                if sentence_active and sentence_texts_raw:
                                    sentence_index += 1
                                    full_sentence_text = "".join(sentence_texts_raw)

                                    if transcription_params.get(
                                        "enable_punctuation_prediction", True
                                    ):
                                        full_sentence_text = await self._apply_final_punctuation_to_sentence(
                                            full_sentence_text, task_id
                                        )

                                    await self._send_sentence_end(
                                        websocket,
                                        task_id,
                                        sentence_index,
                                        audio_time,
                                        full_sentence_text,
                                        sentence_start_time,
                                        enable_itn=transcription_params.get(
                                            "enable_inverse_text_normalization", True
                                        ),
                                    )

                                await self._send_transcription_completed(
                                    websocket, task_id
                                )
                                state = ConnectionState.COMPLETED
                                logger.info(f"[{task_id}] 识别完成")
                                break
                            else:
                                await self._send_task_failed(
                                    websocket, task_id, "Connection not started"
                                )
                        else:
                            await self._send_task_failed(
                                websocket,
                                task_id,
                                f"Invalid message name: {message_name}",
                            )

                    except json.JSONDecodeError as e:
                        logger.error(f"[{task_id}] JSON解析错误: {e}")
                        await self._send_task_failed(
                            websocket, task_id, f"Message Not Json: {message}"
                        )
                    except Exception as e:
                        logger.error(f"[{task_id}] 处理消息异常: {e}")
                        await self._send_task_failed(websocket, task_id, str(e))
                        break

                elif "bytes" in message:
                    if state == ConnectionState.STARTED:
                        audio_bytes = message["bytes"]

                        if not transcription_params:
                            await self._send_task_failed(
                                websocket, task_id, "StartTranscription not received"
                            )
                            continue

                        try:
                            chunk_start_time = audio_time
                            (
                                result_text,
                                result_text_raw,
                                is_sentence_end,
                                audio_cache,
                                audio_time,
                            ) = await self._process_audio_chunk(
                                audio_bytes,
                                audio_cache,
                                punc_cache,
                                transcription_params,
                                audio_time,
                                task_id,
                                is_final=False,
                            )

                            max_empty_count = max(
                                3,
                                (
                                    transcription_params.get(
                                        "max_sentence_silence", 800
                                    )
                                    * 2
                                )
                                // 600,
                            )

                            if not result_text:
                                empty_result_count += 1
                                if (
                                    sentence_active
                                    and empty_result_count >= max_empty_count
                                ):
                                    is_sentence_end = True
                                    logger.debug(
                                        f"[{task_id}] 连续空结果，判断句子结束"
                                    )
                            else:
                                empty_result_count = 0

                            if is_sentence_end and sentence_active:
                                (
                                    flush_result_text,
                                    flush_result_text_raw,
                                    _,
                                    audio_cache,
                                    audio_time,
                                ) = await self._process_audio_chunk(
                                    b"",
                                    audio_cache,
                                    punc_cache,
                                    transcription_params,
                                    audio_time,
                                    task_id,
                                    is_final=True,
                                )

                                if flush_result_text_raw:
                                    if (
                                        not sentence_texts_raw
                                        or flush_result_text_raw
                                        != sentence_texts_raw[-1]
                                    ):
                                        sentence_texts_raw.append(flush_result_text_raw)

                                sentence_index += 1
                                sentence_duration = audio_time - sentence_start_time
                                full_sentence_text = "".join(sentence_texts_raw)

                                if transcription_params.get(
                                    "enable_punctuation_prediction", True
                                ):
                                    full_sentence_text = (
                                        await self._apply_final_punctuation_to_sentence(
                                            full_sentence_text, task_id
                                        )
                                    )

                                logger.debug(
                                    f"[{task_id}] 句子结束 #{sentence_index}: '{full_sentence_text}' "
                                    f"({sentence_duration}ms)"
                                )
                                await self._send_sentence_end(
                                    websocket,
                                    task_id,
                                    sentence_index,
                                    audio_time,
                                    full_sentence_text,
                                    sentence_start_time,
                                    enable_itn=transcription_params.get(
                                        "enable_inverse_text_normalization", True
                                    ),
                                )
                                sentence_active = False
                                sentence_start_time = 0
                                last_sentence_text = ""
                                sentence_texts = []
                                sentence_texts_raw = []
                                empty_result_count = 0
                                audio_cache = {}
                                punc_cache = {}
                            elif result_text:
                                if result_text != last_sentence_text:
                                    last_sentence_text = result_text
                                    if (
                                        not sentence_texts
                                        or result_text != sentence_texts[-1]
                                    ):
                                        sentence_texts.append(result_text)
                                    if (
                                        not sentence_texts_raw
                                        or result_text_raw != sentence_texts_raw[-1]
                                    ):
                                        sentence_texts_raw.append(result_text_raw)

                                    if not sentence_active:
                                        sentence_active = True
                                        sentence_start_time = chunk_start_time
                                        sentence_texts = [result_text]
                                        sentence_texts_raw = [result_text_raw]
                                        empty_result_count = 0
                                        logger.debug(
                                            f"[{task_id}] 句子开始 #{sentence_index + 1}"
                                        )
                                        await self._send_sentence_begin(
                                            websocket,
                                            task_id,
                                            sentence_index + 1,
                                            sentence_start_time,
                                        )

                                    if transcription_params.get(
                                        "enable_intermediate_result", True
                                    ):
                                        await self._send_transcription_result_changed(
                                            websocket,
                                            task_id,
                                            sentence_index + 1,
                                            audio_time,
                                            result_text,
                                        )

                        except Exception as e:
                            logger.error(f"[{task_id}] 音频处理异常: {e}")
                            await self._send_task_failed(
                                websocket, task_id, f"Audio processing failed: {str(e)}"
                            )
                    else:
                        await self._send_task_failed(
                            websocket, task_id, "Connection not started"
                        )

        except WebSocketDisconnect:
            logger.info(f"[{task_id}] 客户端断开连接")
        except Exception as e:
            logger.error(f"[{task_id}] 阿里云WebSocket ASR连接处理异常: {e}")
            try:
                await self._send_task_failed(websocket, task_id, str(e))
            except:
                pass

    def _parse_start_transcription(self, data: dict, task_id: str) -> Optional[dict]:
        """解析StartTranscription消息参数"""
        try:
            payload = data.get("payload", {})

            params = {
                "format": payload.get("format", "pcm"),
                "sample_rate": payload.get("sample_rate", 16000),
                "enable_intermediate_result": payload.get(
                    "enable_intermediate_result", True
                ),
                "enable_punctuation_prediction": payload.get(
                    "enable_punctuation_prediction", True
                ),
                "enable_inverse_text_normalization": payload.get(
                    "enable_inverse_text_normalization", True
                ),
                "max_sentence_silence": payload.get("max_sentence_silence", 800),
                "enable_words": payload.get("enable_words", False),
            }

            logger.info(f"[{task_id}] StartTranscription参数解析成功: {params}")
            return params

        except Exception as e:
            logger.error(f"[{task_id}] 解析StartTranscription失败: {e}")
            return None

    async def _process_audio_chunk(
        self,
        audio_bytes: bytes,
        cache: Dict,
        punc_cache: Dict,
        params: dict,
        current_audio_time: int,
        task_id: str,
        is_final: bool = False,
    ) -> tuple[str, str, bool, Dict, int]:
        """处理音频块，返回带标点文本、无标点文本、是否句子结束、缓存、音频时长"""
        try:
            asr_engine = self._ensure_asr_engine()

            audio_format = params.get("format", "pcm").lower()
            sample_rate_value = params.get("sample_rate", 16000)
            if isinstance(sample_rate_value, (list, tuple)):
                sample_rate_value = sample_rate_value[0] if sample_rate_value else 16000
            if isinstance(sample_rate_value, str):
                sample_rate_value = sample_rate_value.strip()
                if sample_rate_value.isdigit():
                    sample_rate_value = int(sample_rate_value)
                else:
                    raise Exception(f"无效的采样率参数: {sample_rate_value}")
            try:
                sample_rate = int(sample_rate_value)
            except (TypeError, ValueError):
                raise Exception(f"无效的采样率类型: {sample_rate_value}")

            if audio_format == "pcm":
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            elif audio_format == "wav":
                audio_io = io.BytesIO(audio_bytes)
                audio_array, sr = sf.read(audio_io)
                if sr != sample_rate:
                    logger.warning(f"WAV采样率 {sr} 与配置 {sample_rate} 不一致")
            else:
                raise Exception(f"暂不支持的音频格式: {audio_format}")

            audio_array = np.asarray(audio_array, dtype=np.float32)

            chunk_duration_ms = int(len(audio_array) / sample_rate * 1000)
            new_audio_time = current_audio_time + chunk_duration_ms

            chunk_size = [0, 10, 5]
            encoder_chunk_look_back = 4
            decoder_chunk_look_back = 1

            result = asr_engine.realtime_model.generate(
                input=audio_array,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
            )

            result_text_raw = ""
            result_text_with_punc = ""
            is_sentence_end = False

            if result and len(result) > 0:
                result_text_raw = result[0].get("text", "").strip()
                result_text_with_punc = result_text_raw

                # 如果启用实时标点且用户要求标点，则应用实时标点恢复（仅用于中间结果展示）
                if (
                    result_text_raw
                    and settings.ASR_ENABLE_REALTIME_PUNC
                    and params.get("enable_punctuation_prediction", True)
                ):
                    try:
                        from .asr.engine import get_global_punc_realtime_model

                        # 使用全局实时PUNC模型
                        punc_realtime_model = get_global_punc_realtime_model(
                            asr_engine.device
                        )
                        if punc_realtime_model:
                            punc_result = punc_realtime_model.generate(
                                input=result_text_raw, cache=punc_cache
                            )
                            if punc_result and len(punc_result) > 0:
                                result_text_with_punc = (
                                    punc_result[0].get("text", result_text_raw).strip()
                                )
                    except Exception as e:
                        logger.warning(f"[{task_id}] 实时标点恢复失败: {e}")

                if result_text_with_punc and self._is_sentence_boundary(
                    result_text_with_punc
                ):
                    is_sentence_end = True

            if result_text_with_punc:
                logger.debug(f"[{task_id}] 识别: '{result_text_with_punc}'")

            return (
                result_text_with_punc,
                result_text_raw,
                is_sentence_end,
                cache,
                new_audio_time,
            )

        except Exception as e:
            logger.exception(f"[{task_id}] 音频块处理失败: {e}")
            raise e

    async def _apply_final_punctuation_to_sentence(
        self, text: str, task_id: str
    ) -> str:
        """对完整句子应用最终标点恢复（使用离线标点模型添加完整标点包括句末标点）"""
        if not text:
            return text

        try:
            from .asr.engine import get_global_punc_model

            asr_engine = self._ensure_asr_engine()

            # 使用全局PUNC模型
            punc_model = get_global_punc_model(asr_engine.device)

            if punc_model is None:
                logger.info(f"[{task_id}] 标点模型未加载，返回原文本")
                return text

            logger.debug(f"[{task_id}] 应用标点恢复: '{text}'")
            result = punc_model.generate(input=text)

            if result and len(result) > 0:
                punctuated_text = result[0].get("text", text).strip()
                logger.debug(f"[{task_id}] 标点恢复结果: '{punctuated_text}'")
                return punctuated_text
            else:
                return text

        except Exception as e:
            logger.warning(f"[{task_id}] 标点恢复失败: {e}")
            return text

    def _is_sentence_boundary(self, text: str) -> bool:
        """判断是否为句子边界（包含句末标点）"""
        if not text:
            return False
        sentence_endings = ["。", "！", "？", ".", "!", "?", "…"]
        return any(text.endswith(ending) for ending in sentence_endings)

    async def _send_transcription_started(
        self, websocket, task_id: str, session_id: str
    ):
        """发送TranscriptionStarted响应"""
        response = {
            "header": {
                "message_id": AliyunASRWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunASRNamespace.SPEECH_TRANSCRIBER,
                "name": AliyunASRMessageName.TRANSCRIPTION_STARTED,
                "status": AliyunASRStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "session_id": session_id,
            },
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))

    async def _send_sentence_begin(
        self, websocket, task_id: str, index: int, time: int
    ):
        """发送SentenceBegin响应"""
        response = {
            "header": {
                "message_id": AliyunASRWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunASRNamespace.SPEECH_TRANSCRIBER,
                "name": AliyunASRMessageName.SENTENCE_BEGIN,
                "status": AliyunASRStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "index": index,
                "time": time,
            },
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))

    async def _send_transcription_result_changed(
        self, websocket, task_id: str, index: int, time: int, result: str
    ):
        """发送TranscriptionResultChanged响应（中间结果）"""
        response = {
            "header": {
                "message_id": AliyunASRWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunASRNamespace.SPEECH_TRANSCRIBER,
                "name": AliyunASRMessageName.TRANSCRIPTION_RESULT_CHANGED,
                "status": AliyunASRStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "index": index,
                "time": time,
                "result": result,
            },
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))

    async def _send_sentence_end(
        self,
        websocket,
        task_id: str,
        index: int,
        time: int,
        result: str,
        begin_time: int = 0,
        enable_itn: bool = False,
    ):
        """发送SentenceEnd响应"""
        if enable_itn and result:
            logger.debug(f"[{task_id}] 应用ITN: {result}")
            result = apply_itn_to_text(result)
            logger.debug(f"[{task_id}] ITN结果: {result}")

        response = {
            "header": {
                "message_id": AliyunASRWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunASRNamespace.SPEECH_TRANSCRIBER,
                "name": AliyunASRMessageName.SENTENCE_END,
                "status": AliyunASRStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
            "payload": {
                "index": index,
                "time": time,
                "result": result,
                "begin_time": begin_time,
            },
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))

    async def _send_transcription_completed(self, websocket, task_id: str):
        """发送TranscriptionCompleted响应"""
        response = {
            "header": {
                "message_id": AliyunASRWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunASRNamespace.SPEECH_TRANSCRIBER,
                "name": AliyunASRMessageName.TRANSCRIPTION_COMPLETED,
                "status": AliyunASRStatus.SUCCESS,
                "status_message": "GATEWAY|SUCCESS|Success.",
            },
        }
        await websocket.send_text(json.dumps(response, ensure_ascii=False))

    async def _send_task_failed(self, websocket, task_id: str, reason: str):
        """发送TaskFailed响应"""
        response = {
            "header": {
                "namespace": AliyunASRNamespace.SPEECH_TRANSCRIBER,
                "name": AliyunASRMessageName.TASK_FAILED,
                "status": AliyunASRStatus.TASK_FAILED,
                "message_id": AliyunASRWSHeader.generate_message_id(),
                "task_id": task_id,
                "status_text": reason,
            }
        }
        try:
            await websocket.send_text(json.dumps(response, ensure_ascii=False))
            logger.error(f"[{task_id}] 发送TaskFailed: {reason}")
        except:
            pass


# 全局服务实例
_aliyun_websocket_asr_service = None


def get_aliyun_websocket_asr_service() -> AliyunWebSocketASRService:
    """获取阿里云WebSocket ASR服务实例"""
    global _aliyun_websocket_asr_service
    if _aliyun_websocket_asr_service is None:
        _aliyun_websocket_asr_service = AliyunWebSocketASRService()
    return _aliyun_websocket_asr_service
