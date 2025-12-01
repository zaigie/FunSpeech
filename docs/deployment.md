# FunSpeech 部署指南

快速部署 FunSpeech API 服务,支持 CPU 和 GPU 两种模式。

## 🚀 快速部署

### CPU 版本部署

适用于开发测试或无 GPU 环境:

```bash
# 下载配置文件
curl -sSL https://cnb.cool/nexa/FunSpeech/-/git/raw/main/docker-compose.yml -o docker-compose.yml

# 启动服务
docker-compose up -d
```

默认镜像 `docker.cnb.cool/nexa/funspeech:latest` 为 CPU 版本。

### GPU 版本部署

适用于生产环境,提供更快的推理速度:

**前置要求:**
- NVIDIA GPU (CUDA 12.1+)
- 已安装 NVIDIA Container Toolkit

**1. 安装 NVIDIA Container Toolkit**

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**2. 修改 docker-compose.yml**

```yaml
version: "3.8"
services:
  funspeech:
    image: docker.cnb.cool/nexa/funspeech:gpu-latest  # 使用 GPU 镜像
    container_name: funspeech
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/modelscope:/root/.cache/modelscope
      - ./temp:/app/temp
      - ./data:/app/data
      - ./logs:/app/logs
      - ./voices:/app/voices
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**3. 启动服务**

```bash
docker-compose up -d
```

**4. 验证 GPU 可用性**

```bash
# 检查 GPU 是否被识别
docker exec -it funspeech nvidia-smi

# 查看日志确认 GPU 使用
docker-compose logs | grep -i cuda
```

### 数据目录映射

重要数据通过卷映射持久化保存:

| 本地路径              | 容器路径                  | 用途                         | 重要性 |
| --------------------- | ------------------------- | ---------------------------- | ------ |
| `~/.cache/modelscope` | `/root/.cache/modelscope` | 🤖 模型文件缓存              | ⭐⭐⭐ |
| `./temp`              | `/app/temp`               | 📁 临时文件存储              | ⭐⭐   |
| `./data`              | `/app/data`               | 💾 数据库文件(异步 TTS 等)   | ⭐⭐⭐ |
| `./logs`              | `/app/logs`               | 📝 应用日志                  | ⭐⭐   |
| `./voices`            | `/app/voices`             | 🎵 自定义音色                | ⭐⭐⭐ |

> 💡 **提示**: 模型缓存目录非常重要,建议映射到本地以避免重复下载大文件。

## ⚙️ 环境变量配置

### 配置文件 (app/core/config.py)

所有环境变量均在 `app/core/config.py` 中定义,可通过环境变量覆盖默认值。

### 服务器配置

| 环境变量 | 默认值    | 说明                   | 示例        |
| -------- | --------- | ---------------------- | ----------- |
| `HOST`   | `0.0.0.0` | 服务绑定地址           | `127.0.0.1` |
| `PORT`   | `8000`    | 服务端口               | `9000`      |
| `DEBUG`  | `false`   | 开发调试模式           | `true`      |

**使用示例:**

```bash
# 仅监听本地
export HOST=127.0.0.1
export PORT=9000

# 开发模式(启用 API 文档)
export DEBUG=true
```

**影响:**
- `DEBUG=true` 时,API 文档可在 `/docs` 访问
- `DEBUG=false` 时,API 文档自动隐藏

### 日志配置

| 环境变量           | 默认值                       | 说明                     | 可选值                        |
| ------------------ | ---------------------------- | ------------------------ | ----------------------------- |
| `LOG_LEVEL`        | `INFO`                       | 日志级别                 | `DEBUG`, `INFO`, `WARNING`    |
| `LOG_FILE`         | `{BASE_DIR}/logs/funspeech.log` | 日志文件路径             | 任意有效路径                  |
| `LOG_MAX_BYTES`    | `20971520` (20MB)            | 单个日志文件最大大小(字节) | 整数                          |
| `LOG_BACKUP_COUNT` | `50`                         | 日志备份文件数量         | 整数                          |

**使用示例:**

```bash
# 调试模式,输出详细日志
export LOG_LEVEL=DEBUG

# 自定义日志路径和大小
export LOG_FILE=/var/log/funspeech/app.log
export LOG_MAX_BYTES=52428800  # 50MB
export LOG_BACKUP_COUNT=30
```

**影响:**
- `LOG_LEVEL=DEBUG` 输出最详细的日志,包括模型加载、请求处理细节
- 日志文件达到 `LOG_MAX_BYTES` 后自动轮转,保留 `LOG_BACKUP_COUNT` 个备份

### 鉴权配置

| 环境变量  | 默认值 | 说明                              | 示例                |
| --------- | ------ | --------------------------------- | ------------------- |
| `APPTOKEN` | -      | API 访问令牌(X-NLS-Token header)  | `your_secret_token` |
| `APPKEY`  | -      | 应用密钥(appkey 参数)             | `your_app_key`      |

**使用示例:**

```bash
# 启用鉴权
export APPTOKEN=my_secret_token_2024
export APPKEY=my_app_key_2024
```

**影响:**
- **未设置** `APPTOKEN/APPKEY`: 鉴权可选,适合开发环境
- **设置了** `APPTOKEN/APPKEY`: 鉴权必需,所有请求必须提供有效的 token 或 appkey

**请求示例:**

```bash
# 使用 Token
curl -H "X-NLS-Token: my_secret_token_2024" \
  http://localhost:8000/stream/v1/asr

# 使用 AppKey
curl "http://localhost:8000/rest/v1/tts/async?appkey=my_app_key_2024"
```

### 设备配置

| 环境变量     | 默认值 | 说明             | 可选值                  |
| ------------ | ------ | ---------------- | ----------------------- |
| `DEVICE`     | `auto` | ASR 模型设备选择 | `auto`, `cpu`, `cuda:0` |
| `TTS_DEVICE` | `auto` | TTS 模型设备选择 | `auto`, `cpu`, `cuda:0` |

**使用示例:**

```bash
# 强制使用 CPU
export DEVICE=cpu
export TTS_DEVICE=cpu

# 指定 GPU 设备
export DEVICE=cuda:0
export TTS_DEVICE=cuda:1  # 使用第二块 GPU
```

**影响:**
- `auto`: 自动检测,优先使用 GPU(如果可用)
- `cpu`: 强制使用 CPU,降低内存占用但推理较慢
- `cuda:N`: 使用指定的 GPU 设备

### TTS 模型按需加载

| 环境变量       | 默认值 | 说明                     | 可选值                          |
| -------------- | ------ | ------------------------ | ------------------------------- |
| `TTS_MODEL_MODE` | `all`  | TTS 模型加载模式         | `all`, `cosyvoice1`, `cosyvoice2` |

**模式说明:**

| 模式         | 功能           | 磁盘空间 | 内存使用 | 适用场景           |
| ------------ | -------------- | -------- | -------- | ------------------ |
| `all`        | 预设+克隆音色  | ~11GB    | 较高     | 完整功能需求       |
| `cosyvoice1` | 仅预设音色     | ~5.4GB   | 较低     | 标准语音合成       |
| `cosyvoice2` | 仅音色克隆     | ~5.5GB   | 较低     | 个性化音色定制     |

**使用示例:**

```bash
# 仅需预设音色(节省空间)
export TTS_MODEL_MODE=cosyvoice1

# 仅需音色克隆
export TTS_MODEL_MODE=cosyvoice2

# 完整功能
export TTS_MODEL_MODE=all
```

**影响:**
- `cosyvoice1`: 仅加载 CosyVoice-300M-SFT 模型,音色列表仅返回 7 个预设音色
- `cosyvoice2`: 仅加载 CosyVoice2-0.5B 模型,音色列表仅返回克隆音色,支持 `prompt` 参数
- `all`: 加载所有模型,音色列表返回预设+克隆音色

### ASR 模型配置

| 环境变量                | 默认值  | 说明                                 | 可选值                  |
| ----------------------- | ------- | ------------------------------------ | ----------------------- |
| `ASR_MODEL_MODE`        | `all`   | ASR 模型加载模式                     | `realtime`, `offline`, `all` |
| `ASR_ENABLE_REALTIME_PUNC` | `false` | 是否启用实时标点模型(用于中间结果展示) | `true`, `false`         |

**使用示例:**

```bash
# 仅加载实时识别模型
export ASR_MODEL_MODE=realtime

# 启用实时标点
export ASR_ENABLE_REALTIME_PUNC=true
```

**影响:**
- `ASR_MODEL_MODE` 控制加载哪些 ASR 模型
- `ASR_ENABLE_REALTIME_PUNC=true` 会为实时识别中间结果添加标点(增加内存占用)

### 流式ASR远场声音过滤配置

| 环境变量                          | 默认值  | 说明                          | 可选值          |
| --------------------------------- | ------- | ----------------------------- | --------------- |
| `ASR_ENABLE_NEARFIELD_FILTER`     | `true`  | 总开关，是否启用远场声音过滤  | `true`, `false` |
| `ASR_NEARFIELD_RMS_THRESHOLD`     | `0.01`  | RMS能量阈值（宽松模式）       | `0.005`~`0.05`  |
| `ASR_NEARFIELD_FILTER_LOG_ENABLED` | `true`  | 是否记录过滤日志（便于调优）  | `true`, `false` |

**功能说明:**

流式ASR远场声音过滤是一个自动过滤远场声音（如远处说话声、电视人声等环境音）的功能，基于RMS能量阈值检测，零性能开销（<0.1ms），完全可配置。

**使用示例:**

```bash
# 启用远场过滤（默认已启用）
export ASR_ENABLE_NEARFIELD_FILTER=true
export ASR_NEARFIELD_RMS_THRESHOLD=0.01

# 启用调试日志，便于观察过滤效果
export ASR_NEARFIELD_FILTER_LOG_ENABLED=true

# 禁用远场过滤（恢复旧版本行为）
export ASR_ENABLE_NEARFIELD_FILTER=false
```

**影响:**
- 有效减少远场声音和环境音的误触发
- 提升流式识别的准确性和用户体验
- 详细配置和调优指南请参考 [远场过滤文档](./nearfield_filter.md)

### 完整配置示例

**开发环境 (.env.dev):**

```bash
# 服务器配置
HOST=0.0.0.0
PORT=8000
DEBUG=true

# 日志配置
LOG_LEVEL=DEBUG

# 设备配置(使用 CPU 节省资源)
DEVICE=cpu
TTS_DEVICE=cpu

# TTS 模式(仅预设音色)
TTS_MODEL_MODE=cosyvoice1

# 不启用鉴权
# APPTOKEN=
# APPKEY=
```

**生产环境 (.env.prod):**

```bash
# 服务器配置
HOST=0.0.0.0
PORT=8000
DEBUG=false

# 日志配置
LOG_LEVEL=WARNING
LOG_MAX_BYTES=52428800
LOG_BACKUP_COUNT=30

# 设备配置(使用 GPU)
DEVICE=cuda:0
TTS_DEVICE=cuda:0

# TTS 模式(完整功能)
TTS_MODEL_MODE=all

# 启用鉴权
APPTOKEN=your_production_token_here
APPKEY=your_production_appkey_here

# 远场声音过滤（生产环境建议关闭调试日志）
ASR_ENABLE_NEARFIELD_FILTER=true
ASR_NEARFIELD_RMS_THRESHOLD=0.01
ASR_NEARFIELD_FILTER_LOG_ENABLED=false
```

### 使用配置文件

**方式 1: Docker Compose**

```yaml
version: "3.8"
services:
  funspeech:
    image: docker.cnb.cool/nexa/funspeech:latest
    env_file:
      - .env.prod
    # ...
```

**方式 2: 命令行**

```bash
# 使用 .env 文件
docker-compose --env-file .env.prod up -d

# 直接指定环境变量
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG ...
```

## 🔍 服务监控

### 健康检查

```bash
# 检查服务状态
curl http://localhost:8000/stream/v1/asr/health
curl http://localhost:8000/stream/v1/tts/health

# 查看模型状态
curl http://localhost:8000/stream/v1/asr/models

# 查看音色列表
curl http://localhost:8000/stream/v1/tts/voices
```

### 日志监控

```bash
# 实时查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f funspeech

# 查看错误日志
docker-compose logs | grep -i error

# 查看最近100行日志
docker-compose logs --tail=100
```

### 性能监控

```bash
# 容器资源使用情况
docker stats funspeech

# 容器详细信息
docker inspect funspeech

# 磁盘使用情况
du -sh ./data ./logs ./voices ~/.cache/modelscope
```

## 🔧 维护操作

### 更新服务

```bash
# 更新到最新版本
docker-compose pull
docker-compose up -d

# 查看更新日志
docker-compose logs -f
```

### 备份重要数据

```bash
# 备份音色文件
tar -czf voices_backup_$(date +%Y%m%d).tar.gz ./voices/

# 备份数据库
tar -czf data_backup_$(date +%Y%m%d).tar.gz ./data/

# 备份配置文件
cp docker-compose.yml docker-compose.yml.backup
cp .env .env.backup
```

### 清理和重置

```bash
# 清理临时文件
docker exec -it funspeech rm -rf /app/temp/*

# 重启服务
docker-compose restart

# 完全重新部署
docker-compose down
docker-compose up -d
```

## 🚨 故障排除

### 常见问题及解决方案

| 问题             | 症状               | 解决方案                                            |
| ---------------- | ------------------ | --------------------------------------------------- |
| **模型下载失败** | 启动超时、网络错误 | 检查网络,重启容器: `docker-compose restart`        |
| **GPU 内存不足** | CUDA OOM 错误      | 切换 CPU 模式: 设置 `DEVICE=cpu TTS_DEVICE=cpu`     |
| **端口被占用**   | 端口冲突错误       | 修改端口映射: `"8080:8000"`                         |
| **权限错误**     | 文件访问被拒绝     | 修复权限: `sudo chown -R $USER:$USER ./data ./logs` |
| **音色添加失败** | 音色不可用         | 检查文件格式和命名是否正确                          |

### 调试模式

```bash
# 启用调试模式
echo "DEBUG=true" >> .env
docker-compose up -d

# 进入容器调试
docker exec -it funspeech /bin/bash

# 查看详细日志
docker-compose logs -f | grep -E "(ERROR|WARNING|DEBUG)"
```

### 获取支持

如果遇到问题无法解决:

1. **查看日志**: `docker-compose logs -f`
2. **检查配置**: 确认环境变量和文件映射
3. **重启服务**: `docker-compose restart`
4. **提交问题**: 访问项目仓库提交 Issue

## 📊 部署建议

### 资源需求

**最小配置(CPU 版本):**
- CPU: 4 核
- 内存: 8GB
- 磁盘: 20GB

**推荐配置(GPU 版本):**
- CPU: 8 核
- 内存: 16GB
- GPU: NVIDIA GPU (6GB+ 显存)
- 磁盘: 50GB

### 性能优化建议

1. **使用 GPU**: 推理速度提升 5-10 倍
2. **按需加载模型**: 根据实际需求选择 `TTS_MODEL_MODE`
3. **调整日志级别**: 生产环境使用 `LOG_LEVEL=WARNING`
4. **启用模型缓存**: 映射 `~/.cache/modelscope` 避免重复下载

---

🎉 **部署完成!**

访问 `http://localhost:8000/docs`(调试模式下)查看 API 文档,或参考 [README.md](../README.md) 了解详细使用方法。
