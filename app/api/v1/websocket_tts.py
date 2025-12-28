# -*- coding: utf-8 -*-
"""
WebSocket TTS API路由
"""

import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ...services.websocket_tts import get_aliyun_websocket_tts_service

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/ws/v1/tts", tags=["WebSocket TTS"])


@router.websocket("")
async def aliyun_websocket_tts_endpoint(websocket: WebSocket):
    """阿里云WebSocket流式TTS端点"""
    await websocket.accept()
    
    # 获取阿里云WebSocket TTS服务
    service = get_aliyun_websocket_tts_service()
    
    # 生成任务ID
    task_id = f"aliyun_ws_{int(time.time())}_{id(websocket)}"
    
    try:
        # 处理WebSocket连接
        await service._process_websocket_connection(websocket, task_id)
    except WebSocketDisconnect:
        logger.info(f"[{task_id}] 阿里云WebSocket客户端断开连接")
    except Exception as e:
        logger.error(f"[{task_id}] 阿里云WebSocket处理异常: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.get("/test", response_class=HTMLResponse)
async def websocket_test_page():
    """阿里云WebSocket测试页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>阿里云流式语音合成WebSocket测试</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .form-row { display: flex; gap: 20px; margin-bottom: 15px; align-items: end; }
            .form-group { flex: 1; }
            .form-group.small { flex: 0 0 150px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; color: #333; }
            input, textarea, select { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ddd; border-radius: 4px; }
            textarea { resize: vertical; min-height: 60px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; border-radius: 4px; font-size: 14px; }
            button:hover { background: #0056b3; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            button.success { background: #28a745; }
            button.danger { background: #dc3545; }
            .controls { text-align: center; margin: 20px 0; }
            .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
            .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .log { background: #f8f9fa; border: 1px solid #ddd; padding: 15px; height: 400px; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 12px; border-radius: 4px; }
            .log-entry { margin: 2px 0; padding: 2px 5px; border-radius: 3px; }
            .log-entry.info { color: #0c5460; }
            .log-entry.success { color: #155724; }
            .log-entry.error { color: #721c24; background: rgba(248, 215, 218, 0.3); }
            .log-entry.warning { color: #856404; background: rgba(255, 243, 205, 0.3); }
            .audio-container { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 4px; }
            .stats { display: flex; gap: 20px; margin: 15px 0; }
            .stat { background: #e9ecef; padding: 10px; border-radius: 4px; text-align: center; flex: 1; }
            .stat-value { font-size: 18px; font-weight: bold; color: #007bff; }
            .stat-label { font-size: 12px; color: #666; }
            .protocol-info { background: #e7f3ff; padding: 15px; margin: 15px 0; border-radius: 4px; border-left: 4px solid #007bff; }
            
            /* Switch样式 */
            .switch {
                position: relative;
                display: inline-block;
                width: 40px;
                height: 20px;
            }
            
            .switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            
            .slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: .4s;
            }
            
            .slider:before {
                position: absolute;
                content: "";
                height: 16px;
                width: 16px;
                left: 2px;
                bottom: 2px;
                background-color: white;
                transition: .4s;
            }
            
            input:checked + .slider {
                background-color: #007bff;
            }
            
            input:focus + .slider {
                box-shadow: 0 0 1px #007bff;
            }
            
            input:checked + .slider:before {
                transform: translateX(20px);
            }
            
            .slider.round {
                border-radius: 20px;
            }
            
            .slider.round:before {
                border-radius: 50%;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎙️ 阿里云双向流式语音合成测试</h1>
                <p>支持LLM逐词输出场景 - StartSynthesis → 多次RunSynthesis → StopSynthesis</p>
            </div>
            
            <div class="protocol-info">
                <strong>双向流协议说明：</strong>本页面支持在同一WebSocket连接中连续发送多个文本片段进行合成，完美适配LLM逐词输出场景。每次点击"发送文本"都会在当前连接中进行新的合成。
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>WebSocket服务地址:</label>
                    <input type="text" id="wsUrl" value="ws://localhost:8000/ws/v1/tts" />
                </div>
                <div class="form-group small">
                    <label>Token (可选):</label>
                    <input type="password" id="token" placeholder="访问令牌" />
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>当前文本片段:</label>
                    <textarea id="text" placeholder="输入文本片段，支持连续发送多个片段">你好，这是第一个文本片段。</textarea>
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label>语音指令 (prompt，可选):</label>
                    <input type="text" id="prompt" placeholder="例如：用开心的语气说话、请用广东话表达、请慢一点说" />
                    <div style="font-size: 11px; color: #666; margin-top: 3px;">💡 仅对克隆音色生效，可控制语气、方言、语速等</div>
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label>已发送的文本历史:</label>
                    <div id="textHistory" style="background: #f8f9fa; border: 1px solid #ddd; padding: 10px; height: 80px; overflow-y: auto; font-size: 12px; border-radius: 4px;">等待发送文本...</div>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>音色:</label>
                    <select id="voice">
                        <option value="">加载中...</option>
                    </select>
                </div>
                <div class="form-group small">
                    <label>&nbsp;</label>
                    <button onclick="loadVoices()" style="width: 100%; height: 36px; padding: 8px; font-size: 12px;">🔄</button>
                </div>
                <div class="form-group">
                    <label>音频格式:</label>
                    <select id="format">
                        <option value="PCM" selected>PCM (16位)</option>
                        <option value="WAV">WAV</option>
                        <option value="MP3">MP3</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>采样率:</label>
                    <select id="sampleRate">
                        <option value="8000">8000</option>
                        <option value="16000">16000</option>
                        <option value="22050" selected>22050</option>
                        <option value="24000">24000</option>
                    </select>
                </div>
            </div>
            
            <div class="form-row">
                <div class="form-group">
                    <label>语速 (-500 到 500):</label>
                    <input type="range" id="speechRate" min="-500" max="500" value="0" oninput="updateSpeechRateLabel()" />
                    <span id="speechRateLabel">0</span>
                </div>
                <div class="form-group">
                    <label>音量 (0 到 100):</label>
                    <input type="range" id="volume" min="0" max="100" value="50" oninput="updateVolumeLabel()" />
                    <span id="volumeLabel">50</span>
                </div>
                <div class="form-group">
                    <label>句子结束后停止:</label>
                    <div style="display: flex; align-items: center; height: 36px; margin-top: 5px;">
                        <label class="switch">
                            <input type="checkbox" id="autoStopAfterSentence">
                            <span class="slider round"></span>
                        </label>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <button id="connectBtn" onclick="connectWebSocket()">🔌 建立连接</button>
                <button id="sendTextBtn" onclick="sendTextSegment()" disabled>📤 发送文本片段</button>
                <button id="stopBtn" onclick="stopSynthesis()" disabled>🛑 停止合成</button>
                <button onclick="clearLog()">🗑️ 清空日志</button>
                <button onclick="downloadAudio()" id="downloadBtn" disabled>💾 下载音频</button>
            </div>
            
            <div id="status" class="status info">准备就绪</div>
            
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="audioChunks">0</div>
                    <div class="stat-label">音频块数</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="audioSize">0 KB</div>
                    <div class="stat-label">音频大小</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="duration">0.0s</div>
                    <div class="stat-label">合成时长</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="connectionState">未连接</div>
                    <div class="stat-label">连接状态</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="textSegments">0</div>
                    <div class="stat-label">文本片段数</div>
                </div>
            </div>
            
            <div class="audio-container">
                <h3>生成的音频:</h3>
                <audio id="audioPlayer" controls style="width: 100%; margin-bottom: 10px;"></audio>
                <div style="font-size: 12px; color: #666;">
                    💡 提示：PCM格式将自动转换为WAV以便播放和下载
                </div>
            </div>
            
            <div class="form-group">
                <label>实时日志:</label>
                <div id="log" class="log"></div>
            </div>
        </div>

        <script>
            let websocket = null;
            let taskId = null;
            let audioData = new Uint8Array(0);
            let audioChunksCount = 0;
            let startTime = null;
            let isConnected = false;
            let synthParams = null;
            let textSegmentsCount = 0;
            let textHistory = [];
            let connectionState = 'READY'; // READY, STARTED, COMPLETED
            
            // 生成UUID
            function generateUUID() {
                return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                    const r = Math.random() * 16 | 0;
                    const v = c == 'x' ? r : (r & 0x3 | 0x8);
                    return v.toString(16);
                }).replace(/-/g, '').substring(0, 32);
            }
            
            // 生成消息ID
            function generateMessageId() {
                return generateUUID();
            }
            
            // 更新滑块标签
            function updateSpeechRateLabel() {
                document.getElementById('speechRateLabel').textContent = document.getElementById('speechRate').value;
            }
            
            function updateVolumeLabel() {
                document.getElementById('volumeLabel').textContent = document.getElementById('volume').value;
            }
            
            // 日志记录
            function log(message, type = 'info') {
                const logElement = document.getElementById('log');
                const timestamp = new Date().toLocaleTimeString();
                const entry = document.createElement('div');
                entry.className = `log-entry ${type}`;
                entry.textContent = `[${timestamp}] ${message}`;
                logElement.appendChild(entry);
                logElement.scrollTop = logElement.scrollHeight;
            }
            
            // 更新状态
            function updateStatus(message, type = 'info') {
                const statusEl = document.getElementById('status');
                statusEl.textContent = message;
                statusEl.className = `status ${type}`;
            }
            
            // 更新统计信息
            function updateStats() {
                document.getElementById('audioChunks').textContent = audioChunksCount;
                document.getElementById('audioSize').textContent = (audioData.length / 1024).toFixed(1) + ' KB';
                document.getElementById('connectionState').textContent = isConnected ? connectionState : '未连接';
                document.getElementById('textSegments').textContent = textSegmentsCount;
                
                if (startTime) {
                    const duration = (Date.now() - startTime) / 1000;
                    document.getElementById('duration').textContent = duration.toFixed(1) + 's';
                }
            }
            
            // 清空日志
            function clearLog() {
                document.getElementById('log').innerHTML = '';
                audioData = new Uint8Array(0);
                audioChunksCount = 0;
                textSegmentsCount = 0;
                textHistory = [];
                startTime = null;
                updateStats();
                updateTextHistory();
                document.getElementById('audioPlayer').src = '';
                document.getElementById('downloadBtn').disabled = true;
            }
            
            // 更新文本历史显示
            function updateTextHistory() {
                const historyEl = document.getElementById('textHistory');
                if (textHistory.length === 0) {
                    historyEl.textContent = '等待发送文本...';
                } else {
                    historyEl.innerHTML = textHistory.map((text, index) => 
                        `<div style="margin-bottom: 5px; padding: 3px; background: #e9ecef; border-radius: 3px;"><span style="color: #666; font-size: 10px;">[${index + 1}]</span> ${text}</div>`
                    ).join('');
                    historyEl.scrollTop = historyEl.scrollHeight;
                }
            }
            
            // 连接合并数据
            function appendAudioData(newData) {
                const combined = new Uint8Array(audioData.length + newData.length);
                combined.set(audioData);
                combined.set(newData, audioData.length);
                audioData = combined;
                audioChunksCount++;
                updateStats();
            }
            
            // 建立WebSocket连接
            async function connectWebSocket() {
                const wsUrl = document.getElementById('wsUrl').value;
                const token = document.getElementById('token').value;
                
                if (isConnected) {
                    updateStatus('WebSocket已连接', 'info');
                    return;
                }
                
                try {
                    // 重置状态
                    audioData = new Uint8Array(0);
                    audioChunksCount = 0;
                    textSegmentsCount = 0;
                    textHistory = [];
                    startTime = Date.now();
                    taskId = generateUUID();
                    connectionState = 'READY';
                    
                    updateStatus('正在连接WebSocket...', 'info');
                    log('开始连接WebSocket: ' + wsUrl, 'info');
                    
                    // 创建WebSocket连接
                    websocket = new WebSocket(wsUrl);
                    websocket.binaryType = 'arraybuffer';
                    
                    websocket.onopen = async () => {
                        isConnected = true;
                        updateStats();
                        updateTextHistory();
                        updateStatus('WebSocket连接成功，准备发送StartSynthesis', 'success');
                        log('✅ WebSocket连接成功', 'success');
                        
                        // 发送StartSynthesis
                        await sendStartSynthesis();
                    };
                    
                    websocket.onmessage = async (event) => {
                        if (event.data instanceof ArrayBuffer) {
                            // 处理二进制音频数据
                            const audioChunk = new Uint8Array(event.data);
                            appendAudioData(audioChunk);
                            log(`♪ 收到音频数据块 ${audioChunksCount}，大小: ${audioChunk.length} 字节`, 'info');
                        } else {
                            // 处理JSON消息
                            try {
                                const response = JSON.parse(event.data);
                                await handleMessage(response);
                            } catch (e) {
                                log('解析JSON响应失败: ' + e.message, 'error');
                            }
                        }
                    };
                    
                    websocket.onerror = (error) => {
                        log('WebSocket错误: ' + error, 'error');
                        updateStatus('WebSocket连接错误', 'error');
                    };
                    
                    websocket.onclose = () => {
                        isConnected = false;
                        connectionState = 'READY';
                        updateStats();
                        updateStatus('WebSocket连接已关闭', 'info');
                        log('WebSocket连接已关闭', 'info');
                        document.getElementById('connectBtn').disabled = false;
                        document.getElementById('sendTextBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = true;
                        
                        // 如果有音频数据，生成播放文件
                        if (audioData.length > 0) {
                            generateAudioFile();
                        }
                    };
                    
                    document.getElementById('connectBtn').disabled = true;
                    
                } catch (e) {
                    updateStatus('连接失败: ' + e.message, 'error');
                    log('连接失败: ' + e.message, 'error');
                }
            }
            
            // 发送文本片段
            async function sendTextSegment() {
                const text = document.getElementById('text').value;
                
                if (!text.trim()) {
                    updateStatus('请输入文本片段', 'error');
                    return;
                }
                
                if (!isConnected || connectionState !== 'STARTED') {
                    updateStatus('WebSocket未连接或未开始合成', 'error');
                    return;
                }
                
                try {
                    // 记录文本历史
                    textHistory.push(text.trim());
                    textSegmentsCount++;
                    updateTextHistory();
                    updateStats();
                    
                    // 发送RunSynthesis
                    await sendRunSynthesis(text);
                    
                    // 清空输入框，准备下一个片段
                    document.getElementById('text').value = '';
                    document.getElementById('text').placeholder = '输入下一个文本片段...';
                    
                } catch (e) {
                    updateStatus('发送文本失败: ' + e.message, 'error');
                    log('发送文本失败: ' + e.message, 'error');
                }
            }
            
            // 处理消息
            async function handleMessage(response) {
                const header = response.header || {};
                const name = header.name || '';
                const status = header.status || 0;
                
                log(`← 收到消息: ${name} (status: ${status})`, status === 20000000 ? 'success' : 'warning');
                
                switch (name) {
                    case 'SynthesisStarted':
                        if (status === 20000000) {
                            connectionState = 'STARTED';
                            updateStatus('合成已开始，可以发送文本片段', 'success');
                            log('✅ 合成已开始', 'success');
                            updateStats();
                            // 启用发送文本按钮
                            document.getElementById('sendTextBtn').disabled = false;
                            document.getElementById('stopBtn').disabled = false;
                        } else {
                            throw new Error('SynthesisStarted失败: ' + header.status_message);
                        }
                        break;
                        
                    case 'SentenceBegin':
                        updateStatus('句子开始合成', 'info');
                        log('✅ 句子开始', 'success');
                        break;
                        
                    case 'SentenceSynthesis':
                        log('♪ 合成进度更新', 'info');
                        break;
                        
                    case 'SentenceEnd':
                        updateStatus('句子合成结束，可以继续发送文本片段', 'info');
                        log('✅ 句子结束', 'success');
                        break;
                        
                    case 'SynthesisCompleted':
                        connectionState = 'COMPLETED';
                        updateStatus('合成完成！', 'success');
                        log('🎉 合成完成', 'success');
                        updateStats();
                        if (websocket) {
                            websocket.close();
                        }
                        break;
                        
                    case 'TaskFailed':
                        const reason = header.status_text || '未知错误';
                        updateStatus('任务失败: ' + reason, 'error');
                        log('❌ 任务失败: ' + reason, 'error');
                        if (websocket) {
                            websocket.close();
                        }
                        break;
                        
                    default:
                        log(`收到未知消息: ${name}`, 'warning');
                        break;
                }
            }
            
            // 发送StartSynthesis
            async function sendStartSynthesis() {
                const voice = document.getElementById('voice').value;
                const format = document.getElementById('format').value;
                const sampleRate = parseInt(document.getElementById('sampleRate').value);
                const volume = parseInt(document.getElementById('volume').value);
                const speechRate = parseInt(document.getElementById('speechRate').value);
                const autoStopAfterSentence = document.getElementById('autoStopAfterSentence').checked;
                const prompt = document.getElementById('prompt').value;

                const message = {
                    header: {
                        message_id: generateMessageId(),
                        task_id: taskId,
                        namespace: 'FlowingSpeechSynthesizer',
                        name: 'StartSynthesis'
                    },
                    payload: {
                        voice: voice,
                        format: format,
                        sample_rate: sampleRate,
                        volume: volume,
                        speech_rate: speechRate,
                        pitch_rate: 0,
                        enable_subtitle: false,  // 字幕功能已移除
                        prompt: prompt,  // 语音指令控制
                        platform: 'javascript'
                    }
                };

                websocket.send(JSON.stringify(message));
                log('→ 发送StartSynthesis' + (prompt ? ` (prompt: ${prompt})` : ''), 'info');
            }
            
            // 发送RunSynthesis
            async function sendRunSynthesis(text) {
                const message = {
                    header: {
                        message_id: generateMessageId(),
                        task_id: taskId,
                        namespace: 'FlowingSpeechSynthesizer',
                        name: 'RunSynthesis'
                    },
                    payload: {
                        text: text
                    }
                };
                
                websocket.send(JSON.stringify(message));
                log(`→ 发送RunSynthesis: "${text}"`, 'info');
            }
            
            // 停止合成
            async function stopSynthesis() {
                if (websocket && taskId && connectionState === 'STARTED') {
                    const message = {
                        header: {
                            message_id: generateMessageId(),
                            task_id: taskId,
                            namespace: 'FlowingSpeechSynthesizer',
                            name: 'StopSynthesis'
                        }
                    };
                    
                    websocket.send(JSON.stringify(message));
                    log('→ 发送StopSynthesis', 'info');
                    updateStatus('正在停止合成...', 'info');
                    
                    // 禁用发送按钮
                    document.getElementById('sendTextBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = true;
                } else {
                    updateStatus('没有正在进行的合成任务', 'warning');
                }
            }
            
            // 生成音频文件
            function generateAudioFile() {
                if (audioData.length === 0) return;
                
                const format = document.getElementById('format').value;
                let audioBlob;
                let filename;
                
                if (format === 'PCM') {
                    // PCM转WAV
                    const sampleRate = parseInt(document.getElementById('sampleRate').value);
                    const wavData = pcmToWav(audioData, sampleRate);
                    audioBlob = new Blob([wavData], { type: 'audio/wav' });
                    filename = 'synthesis_' + Date.now() + '.wav';
                } else {
                    const mimeType = format === 'WAV' ? 'audio/wav' : 'audio/mpeg';
                    audioBlob = new Blob([audioData], { type: mimeType });
                    filename = 'synthesis_' + Date.now() + '.' + format.toLowerCase();
                }
                
                const audioUrl = URL.createObjectURL(audioBlob);
                document.getElementById('audioPlayer').src = audioUrl;
                document.getElementById('downloadBtn').disabled = false;
                document.getElementById('downloadBtn').onclick = () => downloadFile(audioUrl, filename);
                
                log(`✅ 音频文件已生成: ${filename} (${(audioBlob.size/1024).toFixed(1)} KB)`, 'success');
            }
            
            // PCM转WAV
            function pcmToWav(pcmData, sampleRate) {
                const channels = 1;
                const bitsPerSample = 16;
                const byteRate = sampleRate * channels * bitsPerSample / 8;
                const blockAlign = channels * bitsPerSample / 8;
                const dataSize = pcmData.length;
                const fileSize = 36 + dataSize;
                
                const buffer = new ArrayBuffer(44 + dataSize);
                const view = new DataView(buffer);
                
                // WAV头部
                const writeString = (offset, string) => {
                    for (let i = 0; i < string.length; i++) {
                        view.setUint8(offset + i, string.charCodeAt(i));
                    }
                };
                
                writeString(0, 'RIFF');
                view.setUint32(4, fileSize, true);
                writeString(8, 'WAVE');
                writeString(12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true);
                view.setUint16(22, channels, true);
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, byteRate, true);
                view.setUint16(32, blockAlign, true);
                view.setUint16(34, bitsPerSample, true);
                writeString(36, 'data');
                view.setUint32(40, dataSize, true);
                
                // 拷贝PCM数据
                const pcmView = new Uint8Array(buffer, 44);
                pcmView.set(pcmData);
                
                return buffer;
            }
            
            // 下载文件
            function downloadFile(url, filename) {
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
            
            // 加载音色列表
            async function loadVoices() {
                try {
                    const token = document.getElementById('token').value;
                    const headers = {};
                    
                    if (token) {
                        headers['X-NLS-Token'] = token;
                    }
                    
                    const response = await fetch('/stream/v1/tts/voices', {
                        headers: headers
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        const voiceSelect = document.getElementById('voice');
                        voiceSelect.innerHTML = '';
                        
                        if (data.voices && data.voices.length > 0) {
                            data.voices.forEach(voice => {
                                const option = document.createElement('option');
                                option.value = voice;
                                option.textContent = voice;
                                voiceSelect.appendChild(option);
                            });
                            log(`✅ 加载了 ${data.voices.length} 个音色`, 'success');
                        } else {
                            voiceSelect.innerHTML = '<option value="">暂无可用音色</option>';
                            log('⚠️ 没有找到可用音色', 'warning');
                        }
                    } else if (response.status === 401) {
                        log('⚠️ 需要认证，请在Token字段输入访问令牌', 'warning');
                        document.getElementById('voice').innerHTML = '<option value="">需要认证</option>';
                    } else {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                } catch (error) {
                    log(`❌ 加载音色列表失败: ${error.message}`, 'error');
                    document.getElementById('voice').innerHTML = '<option value="">加载失败</option>';
                }
            }
            
            // 页面加载完成
            window.onload = function() {
                updateStats();
                updateTextHistory();
                loadVoices();
                log('阿里云双向流式WebSocket语音合成测试页面已加载', 'success');
                
                // 回车键快捷发送
                document.getElementById('text').addEventListener('keydown', function(event) {
                    if (event.key === 'Enter' && !event.shiftKey) {
                        event.preventDefault();
                        sendTextSegment();
                    }
                });
            };
            
            // 页面卸载时关闭连接
            window.onbeforeunload = function() {
                if (websocket) {
                    websocket.close();
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)