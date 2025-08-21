# FunSpeech API Server

基于 FunASR 和 CosyVoice 的语音识别与语音合成 API 服务，支持 ASR 和 TTS 功能，接口与阿里云的语音识别和语音合成兼容。

## 项目特点

### ASR 功能

- 🚀 **多模型支持**: 支持 FunASR 和 Dolphin 多种语音识别模型
- 🔄 **动态模型切换**: 通过 `customization_id` 参数动态选择不同的识别模型
- 💾 **智能缓存**: 自动缓存已加载的模型，提高响应速度
- 🌐 **完全兼容阿里云 API**: 支持阿里云语音识别 API 的所有参数和响应格式
- 📊 **实时监控**: 提供模型状态监控和内存使用情况查询
- 📱 **多种输入**: 支持二进制音频流和音频文件链接两种输入方式

### TTS 功能

- 🎵 **统一语音合成**: 基于 CosyVoice 实现高质量语音合成，支持预训练音色和克隆音色
- 🌍 **多语言支持**: 支持中文、英文、日语、韩语、粤语等多种语言
- 🎭 **智能音色管理**: 自动识别音色类型，统一接口调用不同的合成模型
- 🔗 **OpenAI 兼容**: 兼容 OpenAI TTS API 接口格式

### 通用特点

- 📦 **统一架构**: 模块化设计，提高维护性
- 🔧 **灵活配置**: 统一的配置管理系统，支持环境变量和文件配置
- 🛡️ **异常处理**: 完善的错误处理机制，返回标准的错误码
- 🔐 **安全鉴权**: 统一的鉴权体系，支持可选和必需鉴权模式
- 📝 **类型安全**: 完整的 Pydantic 模型定义，确保 API 类型安全

## TODO

- 目前仅兼容阿里云 ASR 接口，大部分参数（format/sample_rate/vocabulary_id）并未实现
- 兼容对齐阿里云语音合成 TTS 接口，目前部分参数还未实现
- ASR 的多模型配置更合理且可扩展化

## 快速开始

### Docker 部署（推荐）

```bash
# 下载 Docker Compose 配置
curl -sSL https://cnb.cool/nexa/FunSpeech/-/git/raw/main/docker-compose.yml -o docker-compose.yml
# 启动服务
docker-compose up -d
```

> 📋 详细部署说明请查看 [DEPLOYMENT.md](./DEPLOYMENT.md)

### 本地开发

#### 环境要求

- Python 3.10+
- CUDA 12.4+ (可选，用于 GPU 加速)
- FFmpeg (用于音频格式转换)

#### 安装运行

```bash
# 拉取本仓库
git clone <repository-url>
cd FunSpeech
git submodule update --init --recursive

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

服务默认运行在 `http://0.0.0.0:8000`

## API 使用

### 端点总览

**ASR（语音识别）:**

- **语音识别**: `POST /stream/v1/asr`
- **模型列表**: `GET /stream/v1/asr/models`
- **健康检查**: `GET /stream/v1/asr/health`

**TTS（语音合成）:**

- **语音合成**: `POST /stream/v1/tts`
- **OpenAI 兼容接口**: `POST /openai/v1/audio/speech`
- **获取音色列表**: `GET /stream/v1/tts/voices`
- **音色详细信息**: `GET /stream/v1/tts/voices/info`
- **刷新音色配置**: `POST /stream/v1/tts/voices/refresh`
- **健康检查**: `GET /stream/v1/tts/health`

**通用:**

- **API 文档**: `GET /docs` (仅在 DEBUG 模式下可用)

### ASR 请求示例

#### 1. 使用默认模型（paraformer-large）

```bash
# 如果设置了XLS_TOKEN环境变量，需要提供正确的token
curl -X POST "http://localhost:8000/stream/v1/asr?appkey=your-appkey&format=wav&sample_rate=16000&enable_punctuation_prediction=true" \
  -H "X-NLS-Token: your_secret_token" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav

# 如果未设置XLS_TOKEN环境变量，X-NLS-Token头部是可选的
curl -X POST "http://localhost:8000/stream/v1/asr?appkey=your-appkey&format=wav&sample_rate=16000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

#### 2. 指定模型（sensevoice-small）

```bash
curl -X POST "http://localhost:8000/stream/v1/asr?appkey=your-appkey&customization_id=sensevoice-small&format=wav" \
  -H "X-NLS-Token: your_secret_token" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

#### 3. 使用 Dolphin 引擎

```bash
curl -X POST "http://localhost:8000/stream/v1/asr?appkey=your-appkey&customization_id=dolphin-small&dolphin_lang_sym=zh&dolphin_region_sym=BEIJING" \
  -H "X-NLS-Token: your_secret_token" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

#### 4. 音频文件链接

```bash
curl -X POST "http://localhost:8000/stream/v1/asr?appkey=your-appkey&customization_id=dolphin-small&audio_address=https://example.com/audio.wav" \
  -H "X-NLS-Token: your_secret_token"
```

#### 5. 查看可用模型

```bash
curl -X GET "http://localhost:8000/stream/v1/asr/models"
```

### TTS 请求示例

#### 1. 语音合成（支持预训练音色和克隆音色）

```bash
# 基础示例（如果设置了XLS_TOKEN环境变量，需要提供X-NLS-Token头部）
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "X-NLS-Token: your_secret_token" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是一个语音合成测试。",
    "voice": "中文女",
    "speech_rate": 0,
    "volume": 50
  }'

# 完整参数示例
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "这是一个包含音量控制的语音合成示例",
    "voice": "中文女",
    "speech_rate": 20,
    "volume": 75,
    "format": "wav",
    "sample_rate": 22050,
    "prompt": "说话温柔一些，语气轻松"
  }'
```

#### 2. OpenAI 兼容接口

```python
from openai import OpenAI

# 如果设置了XLS_TOKEN环境变量，需要提供Bearer token
client = OpenAI(api_key='your_secret_token', base_url='http://localhost:8000/openai/v1')
with client.audio.speech.with_streaming_response.create(
    model='tts-1',
    voice='中文女',
    input='你好，这是OpenAI兼容接口测试。',
    speed=1.0
) as response:
    with open('./test.wav', 'wb') as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

# 如果未设置XLS_TOKEN环境变量，api_key可以是任意值
client = OpenAI(api_key='dummy', base_url='http://localhost:8000/openai/v1')
# ... 其余代码相同
```

```bash
# 使用curl的示例（设置了XLS_TOKEN环境变量时）
curl -X POST "http://localhost:8000/openai/v1/audio/speech" \
  -H "Authorization: Bearer your_secret_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "你好，这是OpenAI兼容接口测试。",
    "voice": "中文女",
    "speed": 1.0,
    "instructions": "说话温柔一些，语气轻松"
  }' \
  --output speech.wav
```

### ASR 支持的参数

| 参数                              | 类型    | 必需 | 默认值           | 描述                                                            |
| --------------------------------- | ------- | ---- | ---------------- | --------------------------------------------------------------- |
| appkey                            | String  | 是   | -                | 应用 Appkey                                                     |
| customization_id                  | String  | 否   | paraformer-large | ASR 模型 ID，可通过 /models 接口查看可用模型                    |
| format                            | String  | 否   | -                | 音频格式 (pcm, wav, opus, speex, amr, mp3, aac, m4a, flac, ogg) |
| sample_rate                       | Integer | 否   | 16000            | 音频采样率 (8000, 16000, 22050, 44100, 48000)                   |
| vocabulary_id                     | String  | 否   | -                | 热词表 (待实现) ID                                              |
| enable_punctuation_prediction     | Boolean | 否   | false            | 是否添加标点                                                    |
| enable_inverse_text_normalization | Boolean | 否   | false            | 中文数字转阿拉伯数字                                            |
| enable_voice_detection            | Boolean | 否   | false            | 是否启用语音检测                                                |
| disfluency                        | Boolean | 否   | false            | 过滤语气 (待实现) 词                                            |
| audio_address                     | String  | 否   | -                | 音频文件下载链接                                                |
| dolphin_lang_sym                  | String  | 否   | zh               | Dolphin 引擎语言符号                                            |
| dolphin_region_sym                | String  | 否   | SHANGHAI         | Dolphin 引擎区域符号                                            |

### TTS 支持的参数

#### 语音合成 (`/stream/v1/tts`)

| 参数        | 类型    | 必需 | 描述                                                                |
| ----------- | ------- | ---- | ------------------------------------------------------------------- |
| text        | String  | 是   | 待合成的文本                                                        |
| format      | String  | 否   | 音频编码格式 (pcm, wav, opus, speex, amr, mp3, aac, m4a, flac, ogg) |
| sample_rate | Integer | 否   | 音频采样率 (8000, 16000, 22050, 44100, 48000)                       |
| voice       | String  | 否   | 音色名称，支持预训练音色（中文女、中文男等）和克隆音色              |
| speech_rate | Float   | 否   | 语速 (-500~500，0 为正常语速，负值为减速，正值为加速)               |
| volume      | Integer | 否   | 音量大小 (0~100，默认值 50)                                         |
| prompt      | String  | 否   | 音色指导文本，用于指导 TTS 模型的音色生成风格                       |

**预训练音色**: 中文女, 中文男, 日语男, 粤语女, 英文女, 英文男, 韩语女  
**克隆音色**: 通过音色管理工具添加的自定义音色

#### OpenAI 兼容接口 (`/openai/v1/audio/speech`)

| 参数            | 类型   | 必需 | 描述                              |
| --------------- | ------ | ---- | --------------------------------- |
| input           | String | 是   | 待合成的文本                      |
| voice           | String | 是   | 音色名称或参考音频路径            |
| speed           | Float  | 否   | 语速 (0.5-2.0)                    |
| model           | String | 否   | 模型名称 (兼容参数，固定为 tts-1) |
| response_format | String | 否   | 响应格式 (固定为 wav)             |
| instructions    | String | 否   | 音色指导文本，等同于 prompt 参数  |

### 支持的音色列表

#### 预训练音色（内置）

- **中文女**: 温柔甜美的中文女性音色
- **中文男**: 深沉稳重的中文男性音色
- **英文女**: 清晰自然的英文女性音色
- **英文男**: 低沉磁性的英文男性音色
- **日语男**: 标准的日语男性音色
- **韩语女**: 清新可爱的韩语女性音色
- **粤语女**: 地道的粤语女性音色

#### 克隆音色（可扩展）

克隆音色需要通过音色管理工具添加，步骤如下：

1. **准备音频和文本文件**：将参考音频文件（`.wav`）和对应的文本文件（`.txt`）放在 `app/services/tts/clone/` 目录下
2. **运行音色管理工具**：`python -m app.services.tts.clone.voice_manager --add <音色名称>`
3. **验证音色可用性**：`GET /stream/v1/tts/voices` 查看音色列表

**注意**：音色名称不能与预训练音色重名，音频文件建议长度为 3-15 秒，音质清晰无噪音。

### Prompt/Instructions 参数说明

#### 功能说明

`prompt`（常规接口）和 `instructions`（OpenAI 兼容接口）参数用于指导 TTS 模型的音色生成风格，两者功能完全相同。

> ⚠️ 目前该参数仅能适用于 **克隆音色**（CosyVoice2），预设音色（SFT）不适用。

#### 使用场景

- **情感控制**: "说话温柔一些" / "语气激动一些" / "说话轻松随意"
- **语速节奏**: "说话慢一点" / "说话节奏明快一些"
- **语调风格**: "用播音腔" / "用朗读的语调" / "像讲故事一样"
- **音色特点**: "声音低沉一些" / "声音甜美一些"

#### 使用建议

1. **简洁明确**: 指导文本应简洁明确，避免过于复杂的描述
2. **中文描述**: 推荐使用中文描述，效果更好
3. **合理长度**: 建议控制在 50 字以内，最长不超过 500 字
4. **适用音色**: 对克隆音色效果更明显，预训练音色也有一定效果

#### 示例

```json
{
  "text": "欢迎来到我们的语音服务平台",
  "voice": "中文女",
  "prompt": "说话温柔一些，像客服一样亲切"
}
```

```json
{
  "input": "今天天气真不错，适合出去走走",
  "voice": "中文男",
  "instructions": "说话轻松自然，像朋友聊天一样"
}
```

## ASR 模型配置

### models.json 配置文件

项目根目录下的 `models.json` 文件用于配置可用的语音识别模型：

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
    },
    "sensevoice-small": {
      "name": "SenseVoice Small",
      "path": "iic/SenseVoiceSmall",
      "engine": "funasr",
      "description": "通用语音识别模型，支持中英文混合识别",
      "languages": ["zh", "en"]
    },
    "dolphin-small": {
      "name": "Dolphin Small",
      "path": "DataoceanAI/dolphin-small",
      "engine": "dolphin",
      "size": "small",
      "description": "轻量级语音识别模型",
      "languages": ["zh", "en"]
    }
  }
}
```

### 配置字段说明

- `name`: 模型显示名称
- `path`: ModelScope Hub 上的模型路径
- `engine`: 引擎类型（funasr 或 dolphin）
- `description`: 模型描述
- `languages`: 支持的语言列表
- `default`: 是否为默认模型
- `size`: Dolphin 模型的大小（small/medium/large）

### 支持的 ASR 模型

#### FunASR 模型

- **Paraformer Large**: 高精度中文语音识别（默认）
- **SenseVoice Small**: 中英文混合识别
- **UniASR 2Pass**: 支持方言的中文识别

#### Dolphin 模型

- **Dolphin Small**: 轻量级模型，适合资源受限环境
- **Dolphin Medium**: 平衡性能与资源消耗
- **Dolphin Large**: 最高精度，适合对准确率要求高的场景

### 响应格式

#### ASR 成功响应

```json
{
  "task_id": "cf7b0c5339244ee29cd4e43fb97f****",
  "result": "北京的天气。",
  "status": 20000000,
  "message": "SUCCESS"
}
```

#### ASR 模型列表响应

```json
{
  "models": [
    {
      "id": "paraformer-large",
      "name": "Paraformer Large",
      "engine": "funasr",
      "description": "高精度中文语音识别模型",
      "languages": ["zh"],
      "default": true,
      "loaded": false,
      "path_exists": true
    }
  ],
  "total": 6,
  "loaded_count": 1
}
```

#### TTS 成功响应

```json
{
  "task_id": "tts_1640995200000_12345678",
  "audio_url": "/tmp/preset_voice_1640995200_1234.wav",
  "status": 20000000,
  "message": "SUCCESS"
}
```

#### 错误响应

```json
{
  "task_id": "8bae3613dfc54ebfa811a17d8a7a****",
  "result": "",
  "status": 40000001,
  "message": "Gateway:ACCESS_DENIED:The token 'c0c1e860f3*******de8091c68a' is invalid!"
}
```

## 鉴权说明

### 环境变量配置

通过环境变量 `XLS_TOKEN` 控制鉴权行为：

- **未设置 XLS_TOKEN**: 鉴权是可选的，客户端可以不提供 token
- **设置了 XLS_TOKEN**: 鉴权是必需的，客户端必须提供正确的 token

```bash
# 启用鉴权
export XLS_TOKEN=your_secret_token_here

# 禁用鉴权（不设置环境变量）
# unset XLS_TOKEN
```

### ASR 接口鉴权

**请求头格式**: `X-NLS-Token: <token>`

```bash
# 必需鉴权时
curl -H "X-NLS-Token: your_secret_token" ...

# 可选鉴权时
curl ...  # 无需提供X-NLS-Token头部
```

### TTS 接口鉴权

**普通 TTS 接口**: 使用 `X-NLS-Token` 头部（与 ASR 相同）

```bash
curl -H "X-NLS-Token: your_secret_token" ...
```

**OpenAI 兼容接口**: 使用 `Authorization: Bearer <token>` 头部

```bash
curl -H "Authorization: Bearer your_secret_token" ...
```

### 鉴权错误响应

当鉴权失败时，返回以下格式的错误：

```json
{
  "task_id": "xxx",
  "result": "",
  "status": 40000001,
  "message": "Gateway:ACCESS_DENIED:The token 'xxxx****' is invalid!"
}
```

## 状态码

| 状态码   | 描述             |
| -------- | ---------------- |
| 20000000 | 请求成功         |
| 40000001 | 身份认证失败     |
| 40000002 | 无效的消息       |
| 40000003 | 无效的参数       |
| 40000004 | 无效的音色参数   |
| 40000005 | 无效的语速参数   |
| 40000006 | 参考音频处理失败 |
| 40000011 | 缺少 appkey      |
| 40000012 | appkey 无效      |
| 40000013 | 参数错误         |
| 40000014 | 不支持的音频格式 |
| 40000015 | 不支持的采样率   |
| 40000021 | 音频数据为空     |
| 40000022 | 音频格式无效     |
| 40000023 | 音频文件过大     |
| 40000024 | 音频下载失败     |
| 41010101 | 不支持的采样率   |
| 50000000 | 内部服务错误     |
| 50000001 | 模型错误         |
| 50000002 | 音频处理失败     |

## 开发说明

### 模型配置

#### ASR 模型 (FunASR & Dolphin)

- **FunASR 模型**: 根据 `models.json` 配置动态加载
- **Dolphin 模型**: 根据 `models.json` 配置动态加载
- **VAD 模型**: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- **标点模型**: `iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch`

#### TTS 模型 (CosyVoice)

- **SFT 模型**: `iic/CosyVoice-300M-SFT` (预训练音色)
- **TTS 模型**: `iic/CosyVoice2-0.5B` (音色克隆)

模型会在首次启动时自动下载，请确保网络连接正常。

### 兼容性说明

- **ASR**: 完全兼容阿里云语音识别 API，支持多模型动态切换，某些高级功能（如热词表）仅提供接口兼容性
- **TTS**: 兼容 OpenAI TTS API 格式，支持多种音色和克隆模式
