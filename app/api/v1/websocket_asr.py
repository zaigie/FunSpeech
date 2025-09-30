# -*- coding: utf-8 -*-
"""
WebSocket ASR API路由
"""

import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ...services.websocket_asr import get_aliyun_websocket_asr_service

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/ws/v1/asr", tags=["WebSocket ASR"])


@router.websocket("")
async def aliyun_websocket_asr_endpoint(websocket: WebSocket):
    """阿里云WebSocket实时ASR端点"""
    await websocket.accept()
    service = get_aliyun_websocket_asr_service()
    task_id = f"aliyun_ws_asr_{int(time.time())}_{id(websocket)}"

    try:
        await service._process_websocket_connection(websocket, task_id)
    except WebSocketDisconnect:
        logger.info(f"[{task_id}] 客户端断开连接")
    except Exception as e:
        logger.error(f"[{task_id}] 连接处理异常: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.get("/test", response_class=HTMLResponse)
async def websocket_asr_test_page():
    """阿里云WebSocket ASR测试页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>阿里云实时语音识别WebSocket测试</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .form-row { display: flex; gap: 20px; margin-bottom: 15px; align-items: end; }
            .form-group { flex: 1; }
            .form-group.small { flex: 0 0 150px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; color: #333; }
            input, select { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; border-radius: 4px; font-size: 14px; margin: 5px; }
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
            .result-container { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 4px; }
            .result-text { font-size: 16px; line-height: 1.8; padding: 15px; background: white; border: 1px solid #ddd; border-radius: 4px; min-height: 150px; max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }
            .stats { display: flex; gap: 20px; margin: 15px 0; }
            .stat { background: #e9ecef; padding: 10px; border-radius: 4px; text-align: center; flex: 1; }
            .stat-value { font-size: 18px; font-weight: bold; color: #007bff; }
            .stat-label { font-size: 12px; color: #666; }
            .protocol-info { background: #e7f3ff; padding: 15px; margin: 15px 0; border-radius: 4px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 阿里云实时语音识别测试</h1>
                <p>WebSocket流式语音识别协议 - StartTranscription → 发送音频流 → StopTranscription</p>
            </div>

            <div class="protocol-info">
                <strong>协议说明：</strong>本页面支持通过麦克风实时录音并进行语音识别。点击"开始识别"后，说话内容将实时转换为文字。
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label>WebSocket服务地址:</label>
                    <input type="text" id="wsUrl" value="ws://localhost:8000/ws/v1/asr" />
                </div>
                <div class="form-group small">
                    <label>Token (可选):</label>
                    <input type="password" id="token" placeholder="访问令牌" />
                </div>
            </div>

            <div class="form-row">
                <div class="form-group">
                    <label>音频格式:</label>
                    <select id="format">
                        <option value="pcm" selected>PCM (16位)</option>
                        <option value="wav">WAV</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>采样率:</label>
                    <select id="sampleRate">
                        <option value="8000">8000</option>
                        <option value="16000" selected>16000</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>中间结果:</label>
                    <select id="enableIntermediate">
                        <option value="true" selected>开启</option>
                        <option value="false">关闭</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>标点预测:</label>
                    <select id="enablePunctuation">
                        <option value="true" selected>开启</option>
                        <option value="false">关闭</option>
                    </select>
                </div>
            </div>

            <div class="controls">
                <button id="startBtn" onclick="startRecognition()" class="success">🎙️ 开始识别</button>
                <button id="stopBtn" onclick="stopRecognition()" disabled class="danger">🛑 停止识别</button>
                <button onclick="clearLog()">🗑️ 清空日志</button>
                <button onclick="clearResult()">🗑️ 清空结果</button>
            </div>

            <div id="status" class="status info">准备就绪，点击"开始识别"按钮开始</div>

            <div id="audioPlaybackContainer" class="result-container" style="display: none;">
                <h3>🔊 录音回放:</h3>
                <audio id="audioPlayback" controls style="width: 100%; margin: 10px 0;"></audio>
                <button onclick="downloadAudio()" style="margin: 5px;">💾 下载录音</button>
                <p style="font-size: 12px; color: #666;">提示: 可以播放刚刚录制的音频,检查录音设备是否正常工作</p>
            </div>

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
                    <div class="stat-label">识别时长</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="connectionState">未连接</div>
                    <div class="stat-label">连接状态</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="sentenceCount">0</div>
                    <div class="stat-label">句子数</div>
                </div>
            </div>

            <div class="result-container">
                <h3>识别结果:</h3>
                <div id="resultText" class="result-text">等待识别...</div>
            </div>

            <div class="form-group">
                <label>实时日志:</label>
                <div id="log" class="log"></div>
            </div>
        </div>

        <script>
            let websocket = null;
            let taskId = null;
            let mediaRecorder = null;
            let audioContext = null;
            let audioChunksCount = 0;
            let audioSizeTotal = 0;
            let startTime = null;
            let isRecording = false;
            let sentenceCount = 0;
            let fullText = "";
            let currentIntermediateText = "";
            let sentences = {};  // 存储各个句子的状态: {index: {text: "", isFinal: false}}
            let recordedAudioChunks = []; // 存储录音数据
            let recordedBlob = null; // 存储录音Blob
            let sendBuffer = []; // 发送缓冲区，用于累积到600ms再发送

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
                document.getElementById('audioSize').textContent = (audioSizeTotal / 1024).toFixed(1) + ' KB';
                document.getElementById('connectionState').textContent = isRecording ? '录音中' : '未连接';
                document.getElementById('sentenceCount').textContent = sentenceCount;

                if (startTime) {
                    const duration = (Date.now() - startTime) / 1000;
                    document.getElementById('duration').textContent = duration.toFixed(1) + 's';
                }
            }

            // 清空日志
            function clearLog() {
                document.getElementById('log').innerHTML = '';
                audioChunksCount = 0;
                audioSizeTotal = 0;
                sentenceCount = 0;
                startTime = null;
                updateStats();
            }

            // 清空结果
            function clearResult() {
                document.getElementById('resultText').textContent = '等待识别...';
                fullText = "";
                currentIntermediateText = "";
                sentences = {};
            }

            // 更新识别结果显示
            function updateResultDisplay(index, text, isFinal = false) {
                // 更新句子状态
                if (!sentences[index]) {
                    sentences[index] = { text: "", isFinal: false };
                }
                sentences[index].text = text;
                sentences[index].isFinal = isFinal;

                // 按index顺序拼接所有句子
                const resultEl = document.getElementById('resultText');
                let displayHtml = "";

                // 获取所有句子的index并排序
                const indices = Object.keys(sentences).map(Number).sort((a, b) => a - b);

                for (let i = 0; i < indices.length; i++) {
                    const idx = indices[i];
                    const sentence = sentences[idx];

                    if (i > 0) {
                        displayHtml += " ";  // 句子之间加空格
                    }

                    if (sentence.isFinal) {
                        // 已完成的句子：黑色
                        displayHtml += sentence.text;
                    } else {
                        // 中间结果：灰色斜体
                        displayHtml += `<span style="color: #999; font-style: italic;">${sentence.text}</span>`;
                    }
                }

                resultEl.innerHTML = displayHtml || "等待识别...";

                // 自动滚动到底部
                resultEl.scrollTop = resultEl.scrollHeight;
            }

            // 开始识别
            async function startRecognition() {
                const wsUrl = document.getElementById('wsUrl').value;
                const token = document.getElementById('token').value;

                if (isRecording) {
                    updateStatus('识别已在进行中', 'warning');
                    return;
                }

                try {
                    // 重置状态
                    audioChunksCount = 0;
                    audioSizeTotal = 0;
                    sentenceCount = 0;
                    fullText = "";
                    currentIntermediateText = "";
                    sentences = {};  // 重置句子状态
                    startTime = Date.now();
                    taskId = generateUUID();

                    updateStatus('正在连接WebSocket...', 'info');
                    log('开始连接WebSocket: ' + wsUrl, 'info');

                    // 创建WebSocket连接
                    websocket = new WebSocket(wsUrl);
                    websocket.binaryType = 'arraybuffer';

                    websocket.onopen = async () => {
                        updateStats();
                        updateStatus('WebSocket连接成功，正在启动麦克风...', 'success');
                        log('✅ WebSocket连接成功', 'success');

                        // 发送StartTranscription
                        await sendStartTranscription();

                        // 启动麦克风录音
                        await startMicrophone();
                    };

                    websocket.onmessage = async (event) => {
                        // 处理文本消息
                        try {
                            const response = JSON.parse(event.data);
                            await handleMessage(response);
                        } catch (e) {
                            log('解析JSON响应失败: ' + e.message, 'error');
                        }
                    };

                    websocket.onerror = (error) => {
                        log('WebSocket错误: ' + error, 'error');
                        updateStatus('WebSocket连接错误', 'error');
                    };

                    websocket.onclose = () => {
                        isRecording = false;
                        updateStats();
                        updateStatus('WebSocket连接已关闭', 'info');
                        log('WebSocket连接已关闭', 'info');
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;

                        // 停止录音
                        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                            mediaRecorder.stop();
                        }
                    };

                    document.getElementById('startBtn').disabled = true;

                } catch (e) {
                    updateStatus('启动失败: ' + e.message, 'error');
                    log('启动失败: ' + e.message, 'error');
                }
            }

            // 停止识别
            async function stopRecognition() {
                if (websocket && taskId) {
                    // 发送缓冲区中剩余的音频数据
                    if (sendBuffer.length > 0) {
                        const currentBufferLength = sendBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
                        const mergedData = new Int16Array(currentBufferLength);
                        let offset = 0;
                        for (const chunk of sendBuffer) {
                            mergedData.set(chunk, offset);
                            offset += chunk.length;
                        }
                        websocket.send(mergedData.buffer);
                        const sampleRate = parseInt(document.getElementById('sampleRate').value);
                        const durationMs = (currentBufferLength / sampleRate * 1000).toFixed(0);
                        log(`📤 发送剩余音频: size=${mergedData.buffer.byteLength}B, samples=${currentBufferLength}, duration=${durationMs}ms`, 'info');
                        sendBuffer = [];
                    }

                    // 停止录音
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                    }

                    // 发送StopTranscription
                    const message = {
                        header: {
                            message_id: generateMessageId(),
                            task_id: taskId,
                            namespace: 'SpeechTranscriber',
                            name: 'StopTranscription'
                        }
                    };

                    websocket.send(JSON.stringify(message));
                    log('→ 发送StopTranscription', 'info');
                    updateStatus('正在停止识别...', 'info');

                    // 禁用停止按钮
                    document.getElementById('stopBtn').disabled = true;
                } else {
                    updateStatus('没有正在进行的识别任务', 'warning');
                }
            }

            // 启动麦克风
            async function startMicrophone() {
                try {
                    const sampleRate = parseInt(document.getElementById('sampleRate').value);

                    // 重置录音数据
                    recordedAudioChunks = [];
                    recordedBlob = null;
                    sendBuffer = []; // 重置发送缓冲区
                    document.getElementById('audioPlaybackContainer').style.display = 'none';

                    // 请求麦克风权限
                    const stream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: sampleRate,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        }
                    });

                    // 创建AudioContext
                    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sampleRate });
                    const source = audioContext.createMediaStreamSource(stream);

                    // 使用ScriptProcessorNode处理音频
                    const bufferSize = 4096;
                    const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

                    // 计算chunk_stride: chunk_size[1] * 960 = 10 * 960 = 9600样本 = 600ms
                    const chunkStride = 9600; // 600ms @ 16kHz
                    log(`音频处理配置: bufferSize=${bufferSize}, chunkStride=${chunkStride}样本 (600ms)`, 'info');

                    processor.onaudioprocess = (event) => {
                        if (websocket && websocket.readyState === WebSocket.OPEN) {
                            const audioData = event.inputBuffer.getChannelData(0);

                            // 计算音频能量
                            let sum = 0;
                            let max = 0;
                            for (let i = 0; i < audioData.length; i++) {
                                const abs = Math.abs(audioData[i]);
                                sum += abs;
                                if (abs > max) max = abs;
                            }
                            const mean = sum / audioData.length;

                            // 转换为16位PCM
                            const pcmData = float32To16BitPCM(audioData);

                            // 保存录音数据（用于回放）
                            recordedAudioChunks.push(new Int16Array(pcmData));

                            // 累积到发送缓冲区
                            sendBuffer.push(new Int16Array(pcmData));
                            const currentBufferLength = sendBuffer.reduce((sum, chunk) => sum + chunk.length, 0);

                            // 当累积到chunkStride时，发送音频数据
                            if (currentBufferLength >= chunkStride) {
                                // 取出chunkStride长度的数据
                                const sendData = new Int16Array(chunkStride);
                                let offset = 0;
                                let remaining = chunkStride;

                                while (remaining > 0 && sendBuffer.length > 0) {
                                    const chunk = sendBuffer[0];
                                    const copyLength = Math.min(remaining, chunk.length);
                                    sendData.set(chunk.subarray(0, copyLength), offset);
                                    offset += copyLength;
                                    remaining -= copyLength;

                                    if (copyLength >= chunk.length) {
                                        // 完全使用了这个chunk，移除它
                                        sendBuffer.shift();
                                    } else {
                                        // 只使用了部分，更新这个chunk
                                        sendBuffer[0] = chunk.subarray(copyLength);
                                    }
                                }

                                // 发送音频数据
                                websocket.send(sendData.buffer);
                                audioChunksCount++;
                                audioSizeTotal += sendData.buffer.byteLength;

                                const durationMs = (chunkStride / sampleRate * 1000).toFixed(0);
                                if ( mean > 0.001 ) {
                                    log(`📤 发送音频块 #${audioChunksCount}: size=${sendData.buffer.byteLength}B, samples=${chunkStride}, duration=${durationMs}ms, max=${max.toFixed(4)}, mean=${mean.toFixed(6)}`, 'info');
                                }
                                // log(`📤 发送音频块 #${audioChunksCount}: size=${sendData.buffer.byteLength}B, samples=${chunkStride}, duration=${durationMs}ms, max=${max.toFixed(4)}, mean=${mean.toFixed(6)}`, mean < 0.001 ? 'warning' : 'info');
                            }

                            updateStats();
                        }
                    };

                    source.connect(processor);
                    processor.connect(audioContext.destination);

                    isRecording = true;
                    document.getElementById('stopBtn').disabled = false;
                    updateStatus('正在识别中...请说话', 'success');
                    log('✅ 麦克风已启动，开始录音', 'success');

                } catch (e) {
                    log('麦克风启动失败: ' + e.message, 'error');
                    updateStatus('麦克风启动失败: ' + e.message, 'error');
                }
            }

            // Float32转16位PCM
            function float32To16BitPCM(float32Array) {
                const buffer = new ArrayBuffer(float32Array.length * 2);
                const view = new DataView(buffer);
                let offset = 0;
                for (let i = 0; i < float32Array.length; i++, offset += 2) {
                    const s = Math.max(-1, Math.min(1, float32Array[i]));
                    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                }
                return new Int16Array(buffer);
            }

            // 处理消息
            async function handleMessage(response) {
                const header = response.header || {};
                const payload = response.payload || {};
                const name = header.name || '';
                const status = header.status || 0;

                // 详细记录消息
                log(`← 收到消息: ${name} (status: ${status})`, status === 20000000 ? 'success' : 'warning');
                if (Object.keys(payload).length > 0) {
                    log(`   payload: ${JSON.stringify(payload)}`, 'info');
                }

                switch (name) {
                    case 'TranscriptionStarted':
                        if (status === 20000000) {
                            updateStatus('识别已开始，正在监听...', 'success');
                            log('✅ 识别已开始', 'success');
                        } else {
                            throw new Error('TranscriptionStarted失败: ' + header.status_message);
                        }
                        break;

                    case 'SentenceBegin':
                        const beginIndex = payload.index ?? '';
                        const beginTime = payload.time ?? 0;
                        log(`🟢 句子开始 #${beginIndex} (time: ${beginTime}ms)`, 'success');
                        log(`   [VAD] 检测到语音活动开始，开始识别新句子`, 'info');
                        break;

                    case 'TranscriptionResultChanged':
                        const intermediateText = payload.result || '';
                        const interimIndex = payload.index ?? 1;
                        const interimTime = payload.time ?? 0;
                        log(`♪ 中间结果 #${interimIndex} (time: ${interimTime}ms): "${intermediateText}"`, 'info');
                        updateResultDisplay(interimIndex, intermediateText, false);
                        break;

                    case 'SentenceEnd':
                        const finalText = payload.result || '';
                        const endIndex = payload.index ?? 1;
                        const endTime = payload.time ?? 0;
                        const beginTimeFromPayload = payload.begin_time ?? 0;
                        const sentenceDuration = endTime - beginTimeFromPayload;

                        sentenceCount++;
                        updateStats();

                        log(`🔴 句子结束 #${endIndex} (time: ${endTime}ms, duration: ${sentenceDuration}ms): "${finalText}"`, 'success');
                        log(`   [VAD] 句子完成: 开始=${beginTimeFromPayload}ms, 结束=${endTime}ms, 时长=${sentenceDuration}ms`, 'info');

                        updateResultDisplay(endIndex, finalText, true);
                        break;

                    case 'TranscriptionCompleted':
                        updateStatus('识别完成！', 'success');
                        log('🎉 识别完成', 'success');
                        updateStats();
                        // 保存并显示录音
                        saveRecordedAudio();
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
                        log(`⚠️ 收到未知消息: ${name}`, 'warning');
                        log(`   完整消息: ${JSON.stringify(response)}`, 'warning');
                        break;
                }
            }

            // 发送StartTranscription
            async function sendStartTranscription() {
                const format = document.getElementById('format').value;
                const sampleRate = parseInt(document.getElementById('sampleRate').value);
                const enableIntermediate = document.getElementById('enableIntermediate').value === 'true';
                const enablePunctuation = document.getElementById('enablePunctuation').value === 'true';

                const message = {
                    header: {
                        message_id: generateMessageId(),
                        task_id: taskId,
                        namespace: 'SpeechTranscriber',
                        name: 'StartTranscription'
                    },
                    payload: {
                        format: format,
                        sample_rate: sampleRate,
                        enable_intermediate_result: enableIntermediate,
                        enable_punctuation_prediction: enablePunctuation,
                        enable_inverse_text_normalization: true
                    }
                };

                websocket.send(JSON.stringify(message));
                log('→ 发送StartTranscription', 'info');
            }

            // 保存录音为WAV格式
            function saveRecordedAudio() {
                if (recordedAudioChunks.length === 0) {
                    log('没有录音数据', 'warning');
                    return;
                }

                try {
                    // 合并所有音频块
                    const totalLength = recordedAudioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
                    const mergedData = new Int16Array(totalLength);
                    let offset = 0;
                    for (const chunk of recordedAudioChunks) {
                        mergedData.set(chunk, offset);
                        offset += chunk.length;
                    }

                    // 创建WAV文件
                    const sampleRate = parseInt(document.getElementById('sampleRate').value);
                    const wavBlob = createWavBlob(mergedData, sampleRate);
                    recordedBlob = wavBlob;

                    // 显示音频播放器
                    const audioPlayback = document.getElementById('audioPlayback');
                    const url = URL.createObjectURL(wavBlob);
                    audioPlayback.src = url;
                    document.getElementById('audioPlaybackContainer').style.display = 'block';

                    log(`✅ 录音已保存 (${(wavBlob.size / 1024).toFixed(1)} KB, ${(totalLength / sampleRate).toFixed(1)}s)`, 'success');
                } catch (e) {
                    log('保存录音失败: ' + e.message, 'error');
                }
            }

            // 创建WAV Blob
            function createWavBlob(pcmData, sampleRate) {
                const numChannels = 1;
                const bitsPerSample = 16;
                const bytesPerSample = bitsPerSample / 8;
                const blockAlign = numChannels * bytesPerSample;
                const byteRate = sampleRate * blockAlign;
                const dataSize = pcmData.length * bytesPerSample;
                const buffer = new ArrayBuffer(44 + dataSize);
                const view = new DataView(buffer);

                // WAV 文件头
                // RIFF chunk descriptor
                writeString(view, 0, 'RIFF');
                view.setUint32(4, 36 + dataSize, true);
                writeString(view, 8, 'WAVE');

                // fmt sub-chunk
                writeString(view, 12, 'fmt ');
                view.setUint32(16, 16, true); // fmt chunk size
                view.setUint16(20, 1, true); // audio format (1 = PCM)
                view.setUint16(22, numChannels, true);
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, byteRate, true);
                view.setUint16(32, blockAlign, true);
                view.setUint16(34, bitsPerSample, true);

                // data sub-chunk
                writeString(view, 36, 'data');
                view.setUint32(40, dataSize, true);

                // 写入PCM数据
                const offset = 44;
                for (let i = 0; i < pcmData.length; i++) {
                    view.setInt16(offset + i * 2, pcmData[i], true);
                }

                return new Blob([buffer], { type: 'audio/wav' });
            }

            // 写入字符串到DataView
            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }

            // 下载录音
            function downloadAudio() {
                if (!recordedBlob) {
                    log('没有可下载的录音', 'warning');
                    return;
                }

                const url = URL.createObjectURL(recordedBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `recording_${new Date().getTime()}.wav`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                log('✅ 录音已下载', 'success');
            }

            // 页面加载完成
            window.onload = function() {
                updateStats();
                log('阿里云实时语音识别WebSocket测试页面已加载', 'success');
            };

            // 页面卸载时关闭连接
            window.onbeforeunload = function() {
                if (websocket) {
                    websocket.close();
                }
                if (audioContext) {
                    audioContext.close();
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
