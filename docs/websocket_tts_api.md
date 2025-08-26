# WebSocket 流式语音合成接口

## 概述

本接口基于阿里云流式语音合成WebSocket协议实现，提供实时流式TTS服务，支持CosyVoice和CosyVoice2模型的流式合成功能。

## 连接信息

- **WebSocket URL**: `ws://localhost:8000/ws/v1/tts`
- **协议**: WebSocket
- **消息格式**: JSON

## 鉴权

支持两种可选的鉴权方式：

1. **Appkey鉴权**: 在请求header中提供appkey
2. **Token鉴权**: 在请求header中提供token

如果服务端未配置相应环境变量（APPKEY、APPTOKEN），则鉴权为可选。

## 请求格式

### 请求消息结构

```json
{
  "header": {
    "appkey": "your_appkey_here",    // 可选，应用密钥
    "token": "your_token_here"       // 可选，访问令牌
  },
  "payload": {
    "text": "待合成的文本内容",           // 必需，待合成文本
    "voice": "中文女",                 // 可选，音色名称
    "format": "wav",                  // 可选，音频格式
    "sample_rate": 22050,            // 可选，采样率
    "volume": 50,                    // 可选，音量 (0-100)
    "speech_rate": 0,                // 可选，语速 (-500 到 500)
    "pitch_rate": 0,                 // 可选，音调 (-500 到 500)
    "enable_subtitle": false         // 可选，是否生成字幕
  }
}
```

### 请求参数说明

#### header (可选)
- `appkey`: 应用密钥，用于API调用认证
- `token`: 访问令牌，用于API调用认证

#### payload (必需)
- `text`: 待合成的文本内容，1-1000字符
- `voice`: 音色名称，支持预设音色和克隆音色
- `format`: 音频格式，支持：pcm, wav, opus, speex, amr, mp3, aac, m4a, flac, ogg
- `sample_rate`: 音频采样率，支持：8000, 16000, 22050, 24000, 44100, 48000
- `volume`: 音量大小，0-100，默认50
- `speech_rate`: 语速倍率，-500到500，0为正常语速
- `pitch_rate`: 音调倍率，-500到500，0为正常音调
- `enable_subtitle`: 是否生成字幕信息，默认false

## 响应格式

### 响应消息结构

```json
{
  "header": {
    "task_id": "task_12345",         // 任务ID
    "event": "started",              // 事件类型
    "status": 20000000,              // 状态码
    "message": "合成开始"             // 状态消息
  },
  "payload": {                       // 响应负载，根据事件类型不同
    // 具体内容见下面的事件类型说明
  }
}
```

### 事件类型

#### 1. started - 合成开始
```json
{
  "header": {
    "task_id": "task_12345",
    "event": "started",
    "status": 20000000,
    "message": "合成开始"
  }
}
```

#### 2. result - 音频数据块
```json
{
  "header": {
    "task_id": "task_12345",
    "event": "result",
    "status": 20000000,
    "message": "音频数据"
  },
  "payload": {
    "audio": "base64编码的音频数据",     // Base64编码的音频字节
    "index": 0,                       // 当前块索引
    "total": 10,                      // 预计总块数
    "timestamp": 1693123456.789       // 时间戳
  }
}
```

#### 3. completed - 合成完成
```json
{
  "header": {
    "task_id": "task_12345",
    "event": "completed",
    "status": 20000000,
    "message": "合成完成"
  },
  "payload": {
    "total": 10,                      // 实际总块数
    "timestamp": 1693123456.789       // 完成时间戳
  }
}
```

#### 4. error/failed - 错误
```json
{
  "header": {
    "task_id": "task_12345",
    "event": "error",
    "status": 50000000,
    "message": "错误描述"
  },
  "payload": {
    "error_code": "50000000",         // 错误代码
    "error_message": "详细错误信息"    // 错误详情
  }
}
```

### 状态码

- `20000000`: 成功
- `40000001`: 参数无效
- `40100005`: 认证失败
- `40300016`: 配额超限
- `50000000`: 内部错误
- `50300018`: 服务不可用

## 音色支持

### 预设音色
- 中文女、中文男
- 英文女、英文男
- 日语男、韩语女、粤语女

### 克隆音色
支持使用预先保存的克隆音色，音色名称为保存时的音色ID。

## 流式合成流程

1. **建立连接**: 客户端连接到WebSocket端点
2. **发送请求**: 发送包含文本和参数的JSON请求
3. **接收started事件**: 服务器确认开始合成
4. **接收result事件**: 持续接收音频数据块（Base64编码）
5. **接收completed事件**: 合成完成
6. **处理音频**: 客户端合并所有音频块并播放/保存

## 技术特性

### CosyVoice流式支持
- **预设音色**: 使用CosyVoice1模型的`stream=True`功能
- **克隆音色**: 使用CosyVoice2模型的`stream=True`功能
- **真实流式**: 音频数据实时生成并推送，无需等待完整合成

### 音频处理
- 音频格式：默认WAV，支持多种格式
- 编码方式：Base64编码传输
- 采样率：自动适配模型（CosyVoice: 22050Hz, CosyVoice2: 24000Hz）

## 客户端示例

### Python客户端
```python
import asyncio
import json
import base64
import websockets

async def tts_client():
    uri = "ws://localhost:8000/ws/v1/tts"
    
    request = {
        "header": {
            "appkey": "your_appkey",
            "token": "your_token"
        },
        "payload": {
            "text": "你好，这是流式语音合成测试！",
            "voice": "中文女",
            "format": "wav",
            "sample_rate": 22050
        }
    }
    
    async with websockets.connect(uri) as websocket:
        # 发送请求
        await websocket.send(json.dumps(request))
        
        audio_chunks = []
        async for message in websocket:
            response = json.loads(message)
            event = response["header"]["event"]
            
            if event == "started":
                print("合成开始")
            elif event == "result":
                audio_data = response["payload"]["audio"]
                audio_chunks.append(audio_data)
                print(f"收到音频块 {response['payload']['index']}")
            elif event == "completed":
                print("合成完成")
                # 合并并保存音频
                combined_audio = "".join(audio_chunks)
                audio_bytes = base64.b64decode(combined_audio)
                with open("output.wav", "wb") as f:
                    f.write(audio_bytes)
                break
            elif event in ["error", "failed"]:
                print(f"合成失败: {response['header']['message']}")
                break

asyncio.run(tts_client())
```

### JavaScript客户端
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/v1/tts');

const request = {
    header: {
        appkey: 'your_appkey',
        token: 'your_token'
    },
    payload: {
        text: '你好，这是流式语音合成测试！',
        voice: '中文女',
        format: 'wav',
        sample_rate: 22050
    }
};

let audioChunks = [];

ws.onopen = function(event) {
    ws.send(JSON.stringify(request));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    const eventType = response.header.event;
    
    if (eventType === 'started') {
        console.log('合成开始');
    } else if (eventType === 'result') {
        audioChunks.push(response.payload.audio);
        console.log(`收到音频块 ${response.payload.index}`);
    } else if (eventType === 'completed') {
        console.log('合成完成');
        // 合并音频并播放
        const combinedAudio = audioChunks.join('');
        const audioBlob = new Blob([Uint8Array.from(atob(combinedAudio), c => c.charCodeAt(0))]);
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
    } else if (['error', 'failed'].includes(eventType)) {
        console.log(`合成失败: ${response.header.message}`);
    }
};
```

## 测试工具

### 1. Web测试页面
访问 `http://localhost:8000/ws/v1/tts/test` 可使用内置的Web测试页面。

### 2. Python测试脚本
项目根目录下的 `test_websocket_tts_client.py` 提供了完整的测试示例。

```bash
python test_websocket_tts_client.py
```

## 性能优化

- 使用真实的流式生成，避免预生成完整音频
- Base64编码优化，减少传输开销
- 自动音色类型检测，选择最优模型
- 内存管理优化，避免大文件缓存

## 错误处理

常见错误及解决方案：

1. **连接失败**: 检查服务器是否运行，端口是否正确
2. **认证失败**: 检查appkey/token是否正确配置
3. **参数错误**: 检查请求参数是否符合规范
4. **音色不存在**: 检查音色名称是否正确，或刷新音色列表
5. **模型未加载**: 检查TTS模型是否正确初始化