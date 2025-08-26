#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云流式语音合成WebSocket测试工具
使用正确的阿里云协议进行测试
"""

import asyncio
import json
import websockets
import uuid
import argparse
import logging
import time
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AliyunTTSTestClient:
    """阿里云TTS测试客户端"""
    
    def __init__(self, ws_url: str, appkey: str = None, token: str = None):
        self.ws_url = ws_url
        self.appkey = appkey
        self.token = token
        self.task_id = None
        self.audio_data = b''
        
    def generate_message_id(self) -> str:
        """生成32位消息ID"""
        return str(uuid.uuid4()).replace('-', '')[:32]
    
    def generate_task_id(self) -> str:
        """生成任务ID"""
        return str(uuid.uuid4()).replace('-', '')[:32]
    
    async def test_streaming_synthesis(self, text: str, voice: str = "中文女", 
                                     format: str = "PCM", sample_rate: int = 22050):
        """测试流式语音合成"""
        self.task_id = self.generate_task_id()
        self.audio_data = b''
        
        logger.info(f"开始测试流式合成，任务ID: {self.task_id}")
        logger.info(f"目标URL: {self.ws_url}")
        logger.info(f"文本: '{text}'")
        logger.info(f"音色: {voice}, 格式: {format}, 采样率: {sample_rate}")
        
        headers = {}
        if self.token:
            headers['X-NLS-Token'] = self.token
            
        try:
            async with websockets.connect(
                self.ws_url, 
                extra_headers=headers,
                ping_interval=None,
                ping_timeout=None
            ) as websocket:
                logger.info("✓ WebSocket连接成功")
                
                # 1. 发送StartSynthesis
                await self._send_start_synthesis(websocket, voice, format, sample_rate)
                
                # 2. 等待SynthesisStarted响应
                await self._wait_for_synthesis_started(websocket)
                
                # 3. 发送RunSynthesis
                await self._send_run_synthesis(websocket, text)
                
                # 4. 接收音频数据和响应
                await self._receive_audio_stream(websocket)
                
                # 5. 发送StopSynthesis
                await self._send_stop_synthesis(websocket)
                
                # 6. 等待SynthesisCompleted响应
                await self._wait_for_synthesis_completed(websocket)
                
                # 7. 保存音频文件
                if self.audio_data:
                    await self._save_audio_file(format, sample_rate)
                
                logger.info("✅ 流式合成测试完成")
                
        except Exception as e:
            logger.error(f"❌ 测试失败: {e}")
            raise
    
    async def _send_start_synthesis(self, websocket, voice: str, format: str, sample_rate: int):
        """发送StartSynthesis消息"""
        message = {
            "header": {
                "message_id": self.generate_message_id(),
                "task_id": self.task_id,
                "namespace": "FlowingSpeechSynthesizer",
                "name": "StartSynthesis",
                "appkey": self.appkey
            },
            "payload": {
                "voice": voice,
                "format": format,
                "sample_rate": sample_rate,
                "volume": 50,
                "speech_rate": 0,
                "pitch_rate": 0,
                "enable_subtitle": False,
                "platform": "python"
            }
        }
        
        await websocket.send(json.dumps(message))
        logger.info("→ 发送StartSynthesis")
    
    async def _wait_for_synthesis_started(self, websocket):
        """等待SynthesisStarted响应"""
        while True:
            response = await websocket.recv()
            try:
                data = json.loads(response)
                header = data.get('header', {})
                
                if header.get('name') == 'SynthesisStarted':
                    if header.get('status') == 20000000:
                        logger.info("✓ 收到SynthesisStarted，合成已开始")
                        return
                    else:
                        raise Exception(f"SynthesisStarted失败: {header.get('status_message')}")
                elif header.get('name') == 'TaskFailed':
                    raise Exception(f"任务失败: {header.get('status_text')}")
                else:
                    logger.info(f"← 收到其他消息: {header.get('name')}")
            except json.JSONDecodeError:
                logger.warning("收到非JSON消息，可能是音频数据")
    
    async def _send_run_synthesis(self, websocket, text: str):
        """发送RunSynthesis消息"""
        message = {
            "header": {
                "message_id": self.generate_message_id(),
                "task_id": self.task_id,
                "namespace": "FlowingSpeechSynthesizer",
                "name": "RunSynthesis"
            },
            "payload": {
                "text": text
            }
        }
        
        await websocket.send(json.dumps(message))
        logger.info(f"→ 发送RunSynthesis: '{text}'")
    
    async def _receive_audio_stream(self, websocket):
        """接收音频流数据"""
        sentence_begin_received = False
        audio_chunks_count = 0
        
        while True:
            try:
                response = await websocket.recv()
                
                # 检查是否为二进制数据（音频）
                if isinstance(response, bytes):
                    self.audio_data += response
                    audio_chunks_count += 1
                    logger.info(f"♪ 收到音频数据块 {audio_chunks_count}，大小: {len(response)} 字节")
                    continue
                
                # 解析JSON响应
                try:
                    data = json.loads(response)
                    header = data.get('header', {})
                    message_name = header.get('name', '')
                    
                    if message_name == 'SentenceBegin':
                        logger.info("✓ 收到SentenceBegin，句子开始")
                        sentence_begin_received = True
                        
                    elif message_name == 'SentenceSynthesis':
                        logger.info("♪ 收到SentenceSynthesis，合成进度")
                        
                    elif message_name == 'SentenceEnd':
                        logger.info("✓ 收到SentenceEnd，句子结束")
                        # 句子结束后可以继续等待更多音频或结束
                        return
                        
                    elif message_name == 'TaskFailed':
                        raise Exception(f"任务失败: {header.get('status_text')}")
                        
                    else:
                        logger.info(f"← 收到消息: {message_name}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"无法解析的响应: {response[:100]}...")
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket连接已关闭")
                break
            except Exception as e:
                logger.error(f"接收数据时出错: {e}")
                break
    
    async def _send_stop_synthesis(self, websocket):
        """发送StopSynthesis消息"""
        message = {
            "header": {
                "message_id": self.generate_message_id(),
                "task_id": self.task_id,
                "namespace": "FlowingSpeechSynthesizer",
                "name": "StopSynthesis"
            }
        }
        
        await websocket.send(json.dumps(message))
        logger.info("→ 发送StopSynthesis")
    
    async def _wait_for_synthesis_completed(self, websocket):
        """等待SynthesisCompleted响应"""
        while True:
            try:
                response = await websocket.recv()
                
                # 可能还有剩余的音频数据
                if isinstance(response, bytes):
                    self.audio_data += response
                    logger.info(f"♪ 收到最后的音频数据块，大小: {len(response)} 字节")
                    continue
                
                try:
                    data = json.loads(response)
                    header = data.get('header', {})
                    
                    if header.get('name') == 'SynthesisCompleted':
                        if header.get('status') == 20000000:
                            logger.info("✅ 收到SynthesisCompleted，合成完成")
                            return
                        else:
                            raise Exception(f"SynthesisCompleted失败: {header.get('status_message')}")
                    elif header.get('name') == 'TaskFailed':
                        raise Exception(f"任务失败: {header.get('status_text')}")
                    else:
                        logger.info(f"← 收到其他消息: {header.get('name')}")
                        
                except json.JSONDecodeError:
                    logger.warning("收到非JSON响应")
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info("WebSocket连接已关闭")
                break
    
    async def _save_audio_file(self, format: str, sample_rate: int):
        """保存音频文件"""
        if not self.audio_data:
            logger.warning("没有音频数据可保存")
            return
            
        timestamp = int(time.time())
        
        if format.upper() == 'PCM':
            # PCM格式：保存为.pcm文件，同时生成wav文件方便播放
            pcm_file = f"test_output_{timestamp}.pcm"
            with open(pcm_file, 'wb') as f:
                f.write(self.audio_data)
            logger.info(f"💾 PCM音频已保存: {pcm_file} ({len(self.audio_data)} 字节)")
            
            # 转换PCM为WAV以便播放
            await self._convert_pcm_to_wav(self.audio_data, sample_rate, f"test_output_{timestamp}.wav")
            
        elif format.upper() == 'WAV':
            # WAV格式：直接保存
            wav_file = f"test_output_{timestamp}.wav"
            with open(wav_file, 'wb') as f:
                f.write(self.audio_data)
            logger.info(f"💾 WAV音频已保存: {wav_file} ({len(self.audio_data)} 字节)")
        
        else:
            # 其他格式
            ext = format.lower()
            audio_file = f"test_output_{timestamp}.{ext}"
            with open(audio_file, 'wb') as f:
                f.write(self.audio_data)
            logger.info(f"💾 音频已保存: {audio_file} ({len(self.audio_data)} 字节)")
    
    async def _convert_pcm_to_wav(self, pcm_data: bytes, sample_rate: int, output_file: str):
        """将PCM数据转换为WAV文件"""
        try:
            import struct
            
            # WAV文件头
            channels = 1  # 单声道
            bits_per_sample = 16
            byte_rate = sample_rate * channels * bits_per_sample // 8
            block_align = channels * bits_per_sample // 8
            data_size = len(pcm_data)
            file_size = 36 + data_size
            
            with open(output_file, 'wb') as f:
                # RIFF头
                f.write(b'RIFF')
                f.write(struct.pack('<L', file_size))
                f.write(b'WAVE')
                
                # fmt块
                f.write(b'fmt ')
                f.write(struct.pack('<L', 16))  # fmt块大小
                f.write(struct.pack('<H', 1))   # PCM格式
                f.write(struct.pack('<H', channels))
                f.write(struct.pack('<L', sample_rate))
                f.write(struct.pack('<L', byte_rate))
                f.write(struct.pack('<H', block_align))
                f.write(struct.pack('<H', bits_per_sample))
                
                # data块
                f.write(b'data')
                f.write(struct.pack('<L', data_size))
                f.write(pcm_data)
                
            logger.info(f"🎵 WAV文件已生成: {output_file}")
            
        except Exception as e:
            logger.error(f"PCM转WAV失败: {e}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='阿里云流式语音合成WebSocket测试工具')
    parser.add_argument('--url', default='ws://localhost:8000/ws/v1/tts', 
                       help='WebSocket服务URL')
    parser.add_argument('--text', default='你好，这是阿里云流式语音合成测试。', 
                       help='待合成的文本')
    parser.add_argument('--voice', default='中文女', 
                       help='音色名称')
    parser.add_argument('--format', choices=['PCM', 'WAV', 'MP3'], default='PCM',
                       help='音频格式')
    parser.add_argument('--sample-rate', type=int, default=22050,
                       choices=[8000, 16000, 22050, 24000],
                       help='采样率')
    parser.add_argument('--appkey', help='阿里云AppKey（可选）')
    parser.add_argument('--token', help='访问Token（可选）')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细日志')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建测试客户端
    client = AliyunTTSTestClient(args.url, args.appkey, args.token)
    
    try:
        # 执行测试
        await client.test_streaming_synthesis(
            text=args.text,
            voice=args.voice,
            format=args.format,
            sample_rate=args.sample_rate
        )
        
        print("\n" + "="*50)
        print("🎉 测试完成！")
        print("="*50)
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))