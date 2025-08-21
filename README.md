# FunSpeech

基于 FunASR 和 CosyVoice 的语音处理 API 服务，提供语音识别（ASR）和语音合成（TTS）功能，与阿里云语音 API 完全兼容。

## ✨ 主要特性

- **🚀 多模型支持** - 集成 FunASR、Dolphin、CosyVoice 等多种高质量模型
- **🌐 完全 API 兼容** - 支持阿里云语音 API 和 OpenAI TTS API 格式
- **🎭 智能音色管理** - 支持预训练音色和自定义克隆音色
- **🔧 灵活配置** - 统一的配置系统，支持环境变量和文件配置
- **🛡️ 安全鉴权** - 完善的身份认证和权限控制
- **💾 性能优化** - 智能模型缓存和动态加载机制

## 📦 快速部署

### Docker 部署（推荐）

```bash
# 下载配置文件
curl -sSL https://cnb.cool/nexa/FunSpeech/-/git/raw/main/docker-compose.yml -o docker-compose.yml

# 启动服务
docker-compose up -d
```

服务将在 `http://localhost:8000` 启动

> 💡 详细部署说明请查看 [DEPLOYMENT.md](./DEPLOYMENT.md)

### 本地开发

**系统要求：**
- Python 3.10+
- CUDA 12.4+（可选，用于 GPU 加速）
- FFmpeg（音频格式转换）

**安装步骤：**

```bash
# 克隆项目
git clone <repository-url>
cd FunSpeech
git submodule update --init --recursive

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

## 🔧 环境配置

### 鉴权配置

```bash
# 启用身份验证
export APPTOKEN=your_secret_token    # Token 验证
export APPKEY=your_app_key           # AppKey 验证

# 开发模式（禁用验证）
# 不设置以上环境变量即可
```

### 配置说明

- **未设置 APPTOKEN/APPKEY**：验证可选，开发模式
- **设置了 APPTOKEN/APPKEY**：验证必需，生产模式

## 📚 API 接口

### ASR（语音识别）

| 端点 | 方法 | 功能 |
|------|------|------|
| `/stream/v1/asr` | POST | 语音识别 |
| `/stream/v1/asr/models` | GET | 模型列表 |
| `/stream/v1/asr/health` | GET | 健康检查 |

### TTS（语音合成）

| 端点 | 方法 | 功能 |
|------|------|------|
| `/stream/v1/tts` | POST | 语音合成 |
| `/openai/v1/audio/speech` | POST | OpenAI 兼容接口 |
| `/stream/v1/tts/voices` | GET | 音色列表 |
| `/stream/v1/tts/voices/info` | GET | 音色详细信息 |
| `/stream/v1/tts/voices/refresh` | POST | 刷新音色配置 |
| `/stream/v1/tts/health` | GET | 健康检查 |

## 🎯 使用示例

### ASR 语音识别

**基础识别（开发模式）：**
```bash
curl -X POST "http://localhost:8000/stream/v1/asr?format=wav&sample_rate=16000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**指定模型识别：**
```bash
curl -X POST "http://localhost:8000/stream/v1/asr?customization_id=sensevoice-small&format=wav" \
  -H "X-NLS-Token: your_token" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**使用音频链接：**
```bash
curl -X POST "http://localhost:8000/stream/v1/asr?audio_address=https://example.com/audio.wav" \
  -H "X-NLS-Token: your_token"
```

### TTS 语音合成

**标准 TTS 接口：**
```bash
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是语音合成测试。",
    "voice": "中文女",
    "speech_rate": 0,
    "volume": 50
  }' \
  --output speech.wav
```

**OpenAI 兼容接口：**
```bash
curl -X POST "http://localhost:8000/openai/v1/audio/speech" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，这是 OpenAI 兼容接口测试。",
    "voice": "中文女",
    "speed": 1.0
  }' \
  --output speech.wav
```

**Python 示例：**
```python
import requests

def text_to_speech(text, voice="中文女", output_file="output.wav"):
    url = "http://localhost:8000/stream/v1/tts"
    
    data = {
        "text": text,
        "voice": voice,
        "format": "wav",
        "sample_rate": 22050,
        "volume": 50
    }
    
    response = requests.post(url, json=data)
    
    if response.headers.get('Content-Type') == 'audio/mpeg':
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"音频已保存至: {output_file}")
    else:
        print("请求失败:", response.json())

# 使用示例
text_to_speech("你好，这是语音合成测试！")
```

## 🎵 音色系统

### 预训练音色
- **中文女** - 温柔甜美的女性音色
- **中文男** - 深沉稳重的男性音色  
- **英文女** - 清晰自然的英文女性音色
- **英文男** - 低沉磁性的英文男性音色
- **日语男** - 标准的日语男性音色
- **韩语女** - 清新可爱的韩语女性音色
- **粤语女** - 地道的粤语女性音色

### 自定义克隆音色

**添加新音色：**
```bash
# 1. 将音频文件 (*.wav) 和文本文件 (*.txt) 放入 app/services/tts/clone/ 目录
# 2. 运行音色管理工具
python -m app.services.tts.clone.voice_manager --add

# 3. 验证音色
curl "http://localhost:8000/stream/v1/tts/voices"
```

**音色管理命令：**
```bash
python -m app.services.tts.clone.voice_manager --list           # 列出所有音色
python -m app.services.tts.clone.voice_manager --remove <名称>  # 删除音色
python -m app.services.tts.clone.voice_manager --info <名称>    # 查看音色信息
python -m app.services.tts.clone.voice_manager --refresh        # 刷新音色列表
```

**音色指导功能：**
```json
{
  "text": "欢迎使用语音服务",
  "voice": "中文女",
  "prompt": "说话温柔一些，像客服一样亲切"
}
```

> ⚠️ 注意：音色指导功能目前仅适用于克隆音色（CosyVoice2）

## ⚙️ 参数配置

### ASR 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `customization_id` | String | paraformer-large | ASR 模型 ID |
| `format` | String | - | 音频格式 (wav, mp3, aac 等) |
| `sample_rate` | Integer | 16000 | 采样率 (8000-48000) |
| `enable_punctuation_prediction` | Boolean | false | 是否添加标点 |
| `enable_inverse_text_normalization` | Boolean | false | 中文数字转换 |
| `audio_address` | String | - | 音频文件链接 |

### TTS 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | String | - | 待合成文本（必需） |
| `voice` | String | - | 音色名称 |
| `format` | String | wav | 音频格式 |
| `sample_rate` | Integer | 22050 | 采样率 |
| `speech_rate` | Float | 0 | 语速 (-500~500) |
| `volume` | Integer | 50 | 音量 (0~100) |
| `prompt` | String | - | 音色指导文本 |

## 🤖 支持的模型

### ASR 模型

**FunASR 系列：**
- **Paraformer Large** - 高精度中文识别（默认）
- **SenseVoice Small** - 高精度多语言混合识别、情感辨识和音频事件检测

**Dolphin 系列：**
- **Dolphin Small** - 多语言、多方言识别模型

### TTS 模型

- **CosyVoice-300M-SFT** - 预训练音色模型
- **CosyVoice2-0.5B** - 音色克隆模型

## 📋 响应格式

### ASR 成功响应
```json
{
  "task_id": "cf7b0c5339244ee29cd4e43fb97f****",
  "result": "识别出的文本内容",
  "status": 20000000,
  "message": "SUCCESS"
}
```

### TTS 成功响应
- **Content-Type**: `audio/mpeg`
- **Headers**: `task_id: tts_1640995200000_12345678`
- **Body**: 音频文件二进制数据

### 错误响应
```json
{
  "task_id": "8bae3613dfc54ebfa811a17d8a7a****",
  "result": "",
  "status": 40000001,
  "message": "Gateway:ACCESS_DENIED:Invalid token"
}
```

## 📊 状态码说明

| 状态码 | 说明 | 解决方案 |
|--------|------|----------|
| 20000000 | 请求成功 | - |
| 40000001 | 身份认证失败 | 检查 token 是否正确 |
| 40000002 | 无效消息 | 检查请求格式 |
| 40000003 | 无效参数 | 检查参数设置 |
| 40000004 | 空闲超时 | 检查网络连接 |
| 40000005 | 请求过多 | 控制并发数量 |
| 50000000 | 服务端错误 | 重试请求 |

## 🛠️ 开发指南

### 模型配置

项目根目录的 `models.json` 文件配置可用的 ASR 模型：

```json
{
  "models": {
    "paraformer-large": {
      "name": "Paraformer Large",
      "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
      "engine": "funasr",
      "description": "高精度中文语音识别模型",
      "languages": ["zh"],
      "default": true
    }
  }
}
```

### API 文档

- **开发模式**：访问 `http://localhost:8000/docs` 查看完整 API 文档
- **生产模式**：API 文档自动隐藏

## 📋 TODO

- [ ] 实现 ASR 热词功能 (vocabulary_id)
- [ ] 实现过滤语气词功能 (disfluency)  
- [ ] 实现 TTS 语调控制 (pitch_rate)
- [ ] 优化多模型配置架构

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目！

## 📞 支持

如有问题或建议，请通过以下方式联系：
- 提交 [Issue](../../issues)
- 查看 [DEPLOYMENT.md](./DEPLOYMENT.md) 部署指南