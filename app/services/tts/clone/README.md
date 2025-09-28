# 零样本音色克隆管理模块

本模块基于 CosyVoice2 官方 API 实现零样本音色克隆和管理功能，使用`add_zero_shot_spk`和`save_spkinfo`实现高效的音色管理。

## 📁 目录结构

```
app/services/tts/clone/
├── README.md                   # 说明文档
├── __init__.py                 # 包初始化文件
└── voice_manager.py           # 音色管理器

项目根目录:
voices/                         # 音色文件目录（用户数据）
├── voice_registry.json        # 音色注册表（自动生成）
├── spk/                        # 音色特征保存目录（自动创建）
│   └── spk2info.pt            # CosyVoice2音色特征文件
├── *.txt                       # 参考文本文件
└── *.wav                       # 参考音频文件
```

**注意：** `voices/` 目录位于项目根目录，包含用户数据，不会被 Git 追踪。

## 🎯 核心特性

- **基于官方 API**: 使用`add_zero_shot_spk`和`save_spkinfo`
- **高效性能**: 音色特征保存在内存中，推理时直接使用 ID 引用
- **统一接口**: 零样本克隆音色与预训练音色使用完全相同的调用方式
- **简化管理**: 音色特征统一保存在`spk2info.pt`文件中

## 🚀 使用方法

### 1. 准备音色文件

在项目根目录的 `voices/` 目录下放置成对的音频和文本文件：

```
voices/
├── 张三.wav    # 参考音频文件
├── 张三.txt    # 对应的参考文本
├── 李四.wav
├── 李四.txt
└── ...
```

**文件要求:**

- 音频格式：WAV 格式，建议采样率 16kHz 或以上
- 音频长度：3-30 秒，内容清晰无杂音
- 文本内容：与音频内容完全一致
- 文件命名：音频和文本使用相同的文件名（不含扩展名）

### 2. 添加音色到模型

使用音色管理器添加音色到模型中：

```bash
# 进入项目根目录
cd /path/to/FunSpeech

# 添加所有音色到模型
python -m app.services.tts.clone.voice_manager --add

# 查看已添加的音色列表
python -m app.services.tts.clone.voice_manager --list

# 查看特定音色信息
python -m app.services.tts.clone.voice_manager --info 张三

# 删除特定音色
python -m app.services.tts.clone.voice_manager --remove 张三

# 刷新音色列表
python -m app.services.tts.clone.voice_manager --refresh
```

### 3. 使用音色进行合成

添加完成后，音色会自动集成到 TTS 系统中，使用方式与预训练音色完全相同：

**API 调用示例:**

```bash
# 使用零样本克隆音色进行语音合成
curl -X POST "http://localhost:8000/tts/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是使用零样本克隆音色的语音合成测试",
    "voice": "张三",
    "speed": 1.0
  }'

# 获取所有可用音色（包括零样本克隆音色）
curl "http://localhost:8000/tts/v1/voices"

# 获取详细音色信息
curl "http://localhost:8000/tts/v1/voices/info"

# 刷新音色配置
curl -X POST "http://localhost:8000/tts/v1/voices/refresh"
```

**Python 调用示例:**

```python
from app.services.tts.engine import get_tts_engine

# 获取TTS引擎
tts_engine = get_tts_engine()

# 查看所有可用音色
voices = tts_engine.get_preset_voices()
print(f"可用音色: {voices}")

# 使用零样本克隆音色合成
output_path = tts_engine.synthesize_speech(
    text="你好，这是零样本克隆音色测试",
    voice="张三",
    speed=1.0
)
print(f"合成完成: {output_path}")

# 直接管理音色
voice_manager = tts_engine.voice_manager
if voice_manager:
    # 添加新音色
    success, total = voice_manager.add_all_voices()
    print(f"添加音色: {success}/{total}")

    # 查看零样本克隆音色
    clone_voices = voice_manager.list_clone_voices()
    print(f"零样本克隆音色: {clone_voices}")
```

## 📝 配置文件说明

### voice_registry.json

音色注册表文件，记录已添加的音色信息：

```json
{
  "version": "2.0",
  "description": "CosyVoice2音色注册表",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T12:00:00",
  "voices": {
    "张三": {
      "name": "张三",
      "reference_text": "这是张三的参考文本内容",
      "audio_file": "张三.wav",
      "text_file": "张三.txt",
      "file_size": 123456,
      "audio_duration": 5.2,
      "added_at": "2024-01-01T12:00:00",
      "status": "active"
    }
  }
}
```

### 音色存储机制

- **内存存储**: 音色特征保存在 `cosyvoice.frontend.spk2info` 中
- **持久化存储**: 保存到 `voices/spk/spk2info.pt` 文件
- **用户数据**: 所有用户相关文件都在 `voices/` 目录中，便于备份和迁移
- **自动加载**: 重新启动时自动加载已保存的音色
- **统一访问**: 通过 `cosyvoice.save_spkinfo()` 保存到自定义路径

## ⚡ 性能特点

- **首次添加**: 需要提取并保存特征到模型中（~100-500ms）
- **后续使用**: 直接通过 ID 引用，无需重复提取（~10-50ms）
- **批量处理**: 多次使用同一音色时性能优异
- **内存效率**: 特征集成在模型内存中，无需额外加载

## 🔧 技术原理

### 官方 API 工作流程

1. **添加音色**: `cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, voice_id)`

   - 提取说话人嵌入向量
   - 提取语音特征和 token
   - 保存到 `frontend.spk2info[voice_id]`

2. **保存音色**: `cosyvoice.save_spkinfo()`

   - 将 `frontend.spk2info` 保存到 `spk2info.pt` 文件

3. **使用音色**: `cosyvoice.inference_zero_shot(text, '', '', zero_shot_spk_id=voice_id)`
   - 直接从 `frontend.spk2info` 获取特征
   - 无需重复提取，快速推理

### 与预训练音色的统一

零样本克隆音色和预训练音色都存储在同一个 `spk2info` 字典中，使用完全相同的调用接口，实现真正的统一管理。

## 🐛 故障排除

### 常见问题

**Q: 音色添加失败**

```
A: 检查音频文件格式和质量，确保文本内容与音频一致
```

**Q: 音色不出现在列表中**

```
A: 执行刷新命令：python -m app.services.tts.clone.voice_manager --refresh
或调用API：POST /tts/v1/voices/refresh
```

**Q: 合成音质不佳**

```
A: 检查原始音频质量，尝试使用更清晰的音频文件重新添加
```

**Q: 性能表现异常**

```
A: 检查项目根目录下voices/spk目录中的spk2info.pt文件是否正常生成
```

### 日志查看

查看详细的运行日志：

```bash
# 查看音色管理日志
python -m app.services.tts.clone.voice_manager --add --verbose

# 查看TTS服务日志
tail -f logs/tts.log
```

## 📚 开发文档

### VoiceManager 类

主要方法：

- `add_voice(voice_name, txt_file, wav_file)`: 添加单个音色
- `add_all_voices()`: 添加所有音色文件对
- `remove_voice(voice_name)`: 移除音色
- `list_voices()`: 列出所有音色
- `list_clone_voices()`: 列出零样本克隆音色
- `is_voice_available(voice_name)`: 检查音色可用性
- `refresh_voices()`: 刷新音色列表

### 集成到 TTS 引擎

TTS 引擎自动检测和使用零样本克隆音色：

```python
# 自动检测零样本克隆音色
if self._voice_manager and voice in self._voice_manager.list_clone_voices():
    return self._synthesize_with_saved_voice(text, voice, speed)
```

## 🎉 总结

零样本音色克隆模块基于 CosyVoice2 的官方设计，提供了简洁高效的音色管理功能。通过使用`add_zero_shot_spk`和`save_spkinfo`，零样本克隆音色与预训练音色享受完全相同的推理性能。
