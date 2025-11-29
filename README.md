<div align="center">

![FunSpeech](./docs/images/banner.png)

  <h3>开箱即用的本地私有化部署语音服务</h3>

基于 FunASR 和 CosyVoice 的语音处理 API 服务,提供语音识别(ASR)和语音合成(TTS)功能,与阿里云语音 API 完全兼容,且支持 Websocket 流式 ASR/TTS 协议。

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Torch-2.3.1-%23EE4C2C?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/CUDA-12.1+-%2376B900?logo=nvidia&logoColor=white)

  <div style="margin: 30px 0;">
    <h3>强劲动力来自</h3>
    <a href="https://cnb.cool" target="_blank">
      <img src="https://docs.cnb.cool/images/logo/svg/LogoCnColorfulIcon.svg" alt="云原生构建" width="120" height="40">
    </a>
  </div>
</div>

## ✨ 主要特性

- **🚀 多模型支持** - 集成 FunASR、Dolphin、CosyVoice 等多种高质量模型
- **🌐 完全 API 兼容** - 支持阿里云语音 API 和 OpenAI TTS API 格式,及 Websocket 流式 ASR/TTS 协议
- **🎭 智能音色管理** - 支持预训练音色和零样本克隆音色
- **🔧 灵活配置** - 统一的配置系统,支持环境变量和文件配置
- **🛡️ 安全鉴权** - 完善的身份认证和权限控制
- **💾 性能优化** - 智能模型缓存和动态加载机制

## 📦 快速部署

### Docker 部署(推荐)

```bash
# 下载配置文件
curl -sSL https://cnb.cool/nexa/FunSpeech/-/git/raw/main/docker-compose.yml -o docker-compose.yml

# 启动服务
docker-compose up -d
```

服务将在 `http://localhost:8000` 启动

**GPU 部署**请将 docker-compose.yml 文件中的 image 替换为 **docker.cnb.cool/nexa/funspeech:gpu-latest**

> 💡 详细部署说明(包括 CPU/GPU 版本区别、环境变量配置)请查看 [部署指南](./docs/deployment.md)

### 数据持久化

FunSpeech 会在以下目录存储持久化数据:

- **`./data`** - 数据库文件(异步 TTS 任务记录等)
- **`./temp`** - 临时文件(音频缓存等)
- **`./logs`** - 日志文件
- **`./voices`** - 零样本音色文件

Docker Compose 已自动配置数据卷映射,确保容器重启后数据不丢失。

对于要使用和下载的模型,您可以在运行中动态下载,也可以提前从 ModelScope 下载后映射,需要的模型在 [支持的模型](#-支持的模型),同时注意提前规划好存储空间以免存储空间不足无法下载～

### 本地开发

**系统要求:**

- Python 3.10+
- CUDA 12.1+(可选,用于 GPU 加速)
- FFmpeg(音频格式转换)

**安装步骤:**

```bash
# 克隆项目
cd FunSpeech
git submodule update --init --recursive

# 安装依赖
pip install -r app/services/tts/third_party/CosyVoice/requirements.txt
pip install -r requirements.txt

# 启动服务
python main.py
```

## 📚 API 接口

### ASR(语音识别)

| 端点                    | 方法 | 功能           |
| ----------------------- | ---- | -------------- |
| `/stream/v1/asr`        | POST | 一句话语音识别 |
| `/stream/v1/asr/models` | GET  | 模型列表       |
| `/stream/v1/asr/health` | GET  | 健康检查       |

**完整接口文档:**

- 一句话 ASR：[阿里云一句话语音识别 API](https://help.aliyun.com/zh/isi/developer-reference/restful-api-2)
- 流式 ASR：[Websocket 协议说明](https://help.aliyun.com/zh/isi/developer-reference/websocket)

**特殊说明:**

- 一句话识别限制音频时长 60 秒
- 热词功能待实现

### TTS(语音合成)

| 端点                            | 方法      | 功能                        |
| ------------------------------- | --------- | --------------------------- |
| `/stream/v1/tts`                | POST      | 语音合成                    |
| `/openai/v1/audio/speech`       | POST      | OpenAI 兼容接口             |
| **`/rest/v1/tts/async`**        | **POST**  | **提交异步语音合成任务** 🚀 |
| **`/rest/v1/tts/async`**        | **GET**   | **查询异步语音合成结果** 🚀 |
| `/stream/v1/tts/voices`         | GET       | 音色列表                    |
| `/stream/v1/tts/voices/info`    | GET       | 音色详细信息                |
| `/stream/v1/tts/voices/refresh` | POST      | 刷新音色配置                |
| `/stream/v1/tts/health`         | GET       | 健康检查                    |
| **`/ws/v1/tts`**                | WebSocket | **双向流式语音合成** 🚀     |
| `/ws/v1/tts/test`               | GET       | WebSocket 测试页面          |

**完整接口文档:**

- 基础 TTS: [语音合成 RESTful API](https://help.aliyun.com/zh/isi/developer-reference/restful-api-3)
- 流式 TTS: [Websocket 协议说明](https://help.aliyun.com/zh/isi/developer-reference/websocket-protocol-description)
- 异步 TTS: [阿里云异步长文本语音合成 RESTful API](https://help.aliyun.com/zh/isi/developer-reference/restful-api)

**特殊说明:**

- 合成传入采样率中，CosyVoice1 采样率固定（默认）为 22050，CosyVoice2 采样率固定（默认）为 24000

## 🎯 快速开始

**ASR 语音识别:**

```bash
curl -X POST "http://localhost:8000/stream/v1/asr?format=wav&sample_rate=16000" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**WebSocket 流式识别测试:** 访问 `http://localhost:8000/ws/v1/asr/test`

**TTS 语音合成:**

```bash
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，这是语音合成测试。", "voice": "中文女"}' \
  --output speech.wav
```

**WebSocket 流式合成测试:** 访问 `http://localhost:8000/ws/v1/tts/test`

> 💡 更多示例请查看 `tests/` 目录或访问 `http://localhost:8000/docs`(开发模式)

## 🎵 音色系统

### 智能音色列表

音色列表 API (`/stream/v1/tts/voices`) 会根据当前的模型模式智能返回对应的音色:

- **cosyvoice1 模式**: 仅返回预设音色列表(7 个)
- **cosyvoice2 模式**: 仅返回零样本克隆音色列表(允许为空)
- **all 模式**: 返回所有音色列表(预设+零样本克隆)

### 预训练音色

- **中文女** - 温柔甜美的女性音色
- **中文男** - 深沉稳重的男性音色
- **英文女** - 清晰自然的英文女性音色
- **英文男** - 低沉磁性的英文男性音色
- **日语男** - 标准的日语男性音色
- **韩语女** - 清新可爱的韩语女性音色
- **粤语女** - 地道的粤语女性音色

### 零样本克隆音色

**准备音色文件:**

克隆音色需要准备一对文件:

- **音频文件** (`*.wav`): 3-30 秒,清晰无噪音,建议 16kHz+ 采样率
- **文本文件** (`*.txt`): 音频对应的文字内容,需完全匹配

文件命名必须一致,例如: `张三.wav` 和 `张三.txt`

**添加新音色:**

```bash
# 1. 将音频和文本文件放入 voices 目录
mkdir -p ./voices
cp 张三.wav 张三.txt ./voices/

# 2. 运行音色管理工具添加
python -m app.services.tts.clone.voice_manager --add

# 3. 验证音色
curl "http://localhost:8000/stream/v1/tts/voices"
```

**音色管理命令:**

```bash
python -m app.services.tts.clone.voice_manager --list           # 列出所有音色
python -m app.services.tts.clone.voice_manager --remove <名称>  # 删除音色
python -m app.services.tts.clone.voice_manager --info <名称>    # 查看音色信息
python -m app.services.tts.clone.voice_manager --refresh        # 刷新音色列表
```

**使用克隆音色:**

```bash
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是使用克隆音色的测试。",
    "voice": "张三"
  }' \
  --output cloned_voice.wav
```

**音色指导功能:**

对于零样本克隆音色,可以使用 `prompt` 参数进行音色指导:

```bash
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "欢迎使用语音服务",
    "voice": "张三",
    "prompt": "说话温柔一些，像客服一样亲切"
  }' \
  --output guided_voice.wav
```

> ⚠️ 注意: 音色指导功能仅适用于零样本克隆音色(CosyVoice2 模型)

## 🤖 支持的模型

服务会在首次使用时自动从 ModelScope 下载模型,也可以提前手动下载以加快启动速度。

### TTS 模型 (语音合成)

通过环境变量 `TTS_MODEL_MODE` 控制加载模式。

| 模型名称               | 加载模式             | 大小  | 说明                              | ModelScope 链接                                         |
| ---------------------- | -------------------- | ----- | --------------------------------- | ------------------------------------------------------- |
| **CosyVoice-300M-SFT** | `cosyvoice1` / `all` | 5.4GB | 预训练音色模型,支持 7 种预设音色  | https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT |
| **CosyVoice2-0.5B**    | `cosyvoice2` / `all` | 5.5GB | 零样本克隆模型,支持音色克隆和指导 | https://www.modelscope.cn/models/iic/CosyVoice2-0.5B    |

**模式说明:**

- `TTS_MODEL_MODE=cosyvoice1` - 仅加载预设音色模型 (~5.4GB)
- `TTS_MODEL_MODE=cosyvoice2` - 仅加载音色克隆模型 (~5.5GB)
- `TTS_MODEL_MODE=all` - 加载全部模型 (~11GB,默认)

### ASR 模型 (语音识别)

通过环境变量 `ASR_MODEL_MODE` 控制加载模式。

| 模型名称                    | 加载模式           | 大小  | 说明                               | ModelScope 链接                                                                                         |
| --------------------------- | ------------------ | ----- | ---------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Paraformer Large (离线)** | `offline` / `all`  | 848MB | 高精度中文离线识别,默认模型        | https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch |
| **Paraformer Large (流式)** | `realtime` / `all` | 848MB | 高精度中文实时流式识别             | https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online  |
| **SenseVoice Small**        | 按需加载           | 897MB | 多语言识别、情感辨识、音频事件检测 | https://www.modelscope.cn/models/iic/SenseVoiceSmall                                                    |
| **Dolphin Small**           | 按需加载           | 600MB | 轻量级多语言识别模型               | https://www.modelscope.cn/models/DataoceanAI/dolphin-small                                              |

**模式说明:**

- `ASR_MODEL_MODE=realtime` - 仅加载实时流式模型 (~848MB)
- `ASR_MODEL_MODE=offline` - 仅加载离线模型 (~848MB,默认 Paraformer Large)
- `ASR_MODEL_MODE=all` - 加载全部模型 (~1.7GB,包含离线+流式)

**自定义模型预加载:**

默认情况下，Paraformer Large 会在启动时自动加载。如果需要在启动时预加载其他模型（如 SenseVoice、Dolphin），可以使用 `AUTO_LOAD_CUSTOM_ASR_MODELS` 环境变量：

```bash
# 预加载单个自定义模型
export AUTO_LOAD_CUSTOM_ASR_MODELS="sensevoice-small"

# 预加载多个自定义模型（逗号分隔）
export AUTO_LOAD_CUSTOM_ASR_MODELS="sensevoice-small,dolphin-small"
```

这样在启动时就会自动下载并加载指定的模型，避免首次调用时的等待时间。模型配置详见 `app/services/asr/models.json`。

### 辅助模型

| 模型名称             | 类型       | 大小  | 说明                                                                    | ModelScope 链接                                                                                |
| -------------------- | ---------- | ----- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **PUNC Transformer** | 标点预测   | 283MB | 为离线识别结果添加标点符号                                              | https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch      |
| **PUNC Realtime**    | 实时标点   | 279MB | 为实时识别中间结果添加标点(可选,需设置 `ASR_ENABLE_REALTIME_PUNC=true`) | https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 |
| **FSMN VAD**         | 语音检测   | 3.9MB | 检测语音片段,过滤静音和噪音                                             | https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch                  |
| **CAM++ Speaker**    | 说话人识别 | 28MB  | 说话人特征提取(未启用)                                                  | https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common                       |

### 提前下载模型

**安装 ModelScope CLI:**

```bash
pip install modelscope
```

**下载 TTS 模型:**

```bash
# 预设音色模型 (TTS_MODEL_MODE=cosyvoice1 或 all)
modelscope download --model iic/CosyVoice-300M-SFT

# 音色克隆模型 (TTS_MODEL_MODE=cosyvoice2 或 all)
modelscope download --model iic/CosyVoice2-0.5B
```

**下载 ASR 模型:**

```bash
# 离线模型 (ASR_MODEL_MODE=offline 或 all)
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
modelscope download --model iic/SenseVoiceSmall

# 流式模型 (ASR_MODEL_MODE=realtime 或 all)
modelscope download --model iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
```

**下载辅助模型(按需):**

```bash
# 标点预测模型(离线识别使用)
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch

# 实时标点模型(实时识别使用,可选)
modelscope download --model iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727

# VAD 模型
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
```

> 💡 **提示**: 模型默认下载到 `~/.cache/modelscope/hub`,Docker 部署时需映射此目录以复用模型文件。

### 存储空间规划

根据使用场景规划所需存储空间:

| 场景           | 环境变量配置                                             | 所需模型                   | 总大小 |
| -------------- | -------------------------------------------------------- | -------------------------- | ------ |
| **最小部署**   | `TTS_MODEL_MODE=cosyvoice1`<br>`ASR_MODEL_MODE=offline`  | 1 个 TTS + 离线 ASR + 辅助 | ~7GB   |
| **实时流式**   | `TTS_MODEL_MODE=cosyvoice1`<br>`ASR_MODEL_MODE=realtime` | 1 个 TTS + 流式 ASR + 辅助 | ~7GB   |
| **完整 TTS**   | `TTS_MODEL_MODE=all`<br>`ASR_MODEL_MODE=offline`         | 2 个 TTS + 离线 ASR + 辅助 | ~12GB  |
| **全功能部署** | `TTS_MODEL_MODE=all`<br>`ASR_MODEL_MODE=all`             | 全部模型                   | ~14GB  |

### API 文档

- **开发模式**: 访问 `http://localhost:8000/docs` 查看完整 API 文档
- **生产模式**: API 文档自动隐藏

## 🌐 相关链接

- **部署指南**: [详细文档](./docs/deployment.md)
- **CosyVoice 模型**: [CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice)
- **Dolphin 模型**: [DataoceanAI/Dolphin](https://github.com/DataoceanAI/Dolphin)
- **FunASR**: [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)

## 📋 TODO

- [ ] 实现 ASR 热词功能 (vocabulary_id)
- [ ] 实现过滤语气词功能 (disfluency)
- [ ] 实现 TTS 语调控制 (pitch_rate)
- [ ] 实现长录音文件异步识别接口

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目!
