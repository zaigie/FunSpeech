# FunSpeech 部署指南

一站式 Docker 部署指南，让您快速启动 FunSpeech API 服务。

## 🚀 一键部署

### Docker Compose（推荐）

最简单的部署方式，适合大多数用户：

```bash
# 下载配置文件并启动
curl -sSL https://cnb.cool/nexa/FunSpeech/-/git/raw/main/docker-compose.yml -o docker-compose.yml
docker-compose up -d
```

服务将在 `http://localhost:8000` 启动，首次启动需要下载模型，请耐心等待。

### 使用预构建镜像

如果需要自定义配置，可以直接使用 Docker 镜像：

```bash
docker run -d \
  --name funspeech \
  -p 8000:8000 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  -v $(pwd)/data:/app/temp \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/voices:/app/voices \
  docker.cnb.cool/nexa/funspeech:latest
```

## ⚙️ 配置选项

### 环境变量配置

通过环境变量自定义服务行为：

| 变量             | 默认值    | 说明                 | 示例                       |
| ---------------- | --------- | -------------------- | -------------------------- |
| `HOST`           | `0.0.0.0` | 服务绑定地址         | `127.0.0.1`                |
| `PORT`           | `8000`    | 服务端口             | `9000`                     |
| `DEBUG`          | `false`   | 开发调试模式         | `true`                     |
| `LOG_LEVEL`      | `INFO`    | 日志级别             | `DEBUG`, `WARNING`         |
| `DEVICE`         | `auto`    | ASR 设备选择         | `cpu`, `cuda:0`            |
| `TTS_DEVICE`     | `auto`    | TTS 设备选择         | `cpu`, `cuda:0`            |
| `TTS_MODEL_MODE` | `all`     | TTS 模型按需加载模式 | `cosyvoice1`, `cosyvoice2` |
| `APPTOKEN`       | -         | API 访问令牌         | `your_secret_token`        |
| `APPKEY`         | -         | 应用密钥             | `your_app_key`             |

**配置示例：**

```bash
# 创建环境变量文件
cat > .env << EOF
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO
DEVICE=auto
TTS_DEVICE=auto
TTS_MODEL_MODE=all
APPTOKEN=your_secret_token
APPKEY=your_app_key
EOF

# 使用环境变量启动
docker-compose --env-file .env up -d
```

### 数据目录映射

重要数据通过卷映射持久化保存：

| 本地路径              | 容器路径                  | 用途            | 重要性 |
| --------------------- | ------------------------- | --------------- | ------ |
| `~/.cache/modelscope` | `/root/.cache/modelscope` | 🤖 模型文件缓存 | ⭐⭐⭐ |
| `./data`              | `/app/temp`               | 📁 临时文件存储 | ⭐⭐   |
| `./logs`              | `/app/logs`               | 📝 应用日志     | ⭐⭐   |
| `./voices`            | `/app/voices`             | 🎵 自定义音色   | ⭐⭐⭐ |

> 💡 **提示**：模型缓存目录非常重要，建议映射到本地以避免重复下载大文件。

## 🎮 GPU 加速配置

### 安装 NVIDIA 容器工具包

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 启用 GPU 支持

修改 `docker-compose.yml` 文件：

```yaml
version: "3.8"
services:
  funspeech:
    image: docker.cnb.cool/nexa/funspeech:gpu-latest
    # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 验证 GPU 可用性

```bash
# 检查 GPU 是否被识别
docker exec -it funspeech nvidia-smi

# 查看服务日志确认 GPU 使用
docker-compose logs | grep -i cuda
```

## 🎵 音色管理系统

### 添加自定义音色

**步骤 1：准备音色文件**

```bash
# 创建音色目录
mkdir -p ./voices

# 准备音色文件（示例：张三的音色）
# 张三.wav - 音频文件（3-30秒，清晰无噪音）
# 张三.txt - 对应文本内容
```

**步骤 2：添加到系统**

```bash
# 将文件复制到映射目录
cp 张三.wav 张三.txt ./voices/

# 进入容器添加音色
docker exec -it funspeech python -m app.services.tts.clone.voice_manager --add
```

**步骤 3：验证和使用**

```bash
# 查看所有音色
docker exec -it funspeech python -m app.services.tts.clone.voice_manager --list

# 测试新音色
curl -X POST "http://localhost:8000/stream/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是张三的声音测试。",
    "voice": "张三"
  }' \
  --output test_voice.wav
```

### 音色文件标准

| 要求项       | 规范     | 说明                     |
| ------------ | -------- | ------------------------ |
| **音频格式** | WAV      | 建议 16kHz+ 采样率       |
| **音频长度** | 3-30 秒  | 太短效果差，太长训练慢   |
| **音频质量** | 高质量   | 无背景噪音、回音         |
| **文本匹配** | 完全一致 | 音频内容与文本完全对应   |
| **文件命名** | 统一前缀 | `name.wav` 和 `name.txt` |

### 音色管理命令

```bash
# 进入容器后可用的管理命令
docker exec -it funspeech python -m app.services.tts.clone.voice_manager \
  --list                    # 查看所有音色
  --list-clone             # 仅查看零样本克隆音色
  --add                    # 添加新音色
  --remove <音色名>         # 删除指定音色
  --info <音色名>           # 查看音色详细信息
  --refresh                # 刷新音色列表
  --registry-info          # 查看注册表信息
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

## 🎯 TTS 模型按需加载优化

### 模型模式选择

针对不同使用场景选择合适的模型模式以优化资源使用：

```bash
# 仅需预设音色场景（推荐：轻量部署）
export TTS_MODEL_MODE=cosyvoice1

# 仅需音色克隆场景（推荐：个性化应用）
export TTS_MODEL_MODE=cosyvoice2

# 需要完整功能场景（推荐：全功能部署）
export TTS_MODEL_MODE=all
```

### 资源使用对比

| 模式       | 磁盘空间 | 内存使用 | 启动时间 | 适用场景       |
| ---------- | -------- | -------- | -------- | -------------- |
| cosyvoice1 | ~5.4GB   | 较低     | 较快     | 标准语音合成   |
| cosyvoice2 | ~5.5GB   | 较低     | 较快     | 个性化音色定制 |
| all        | ~11GB    | 较高     | 较慢     | 完整功能需求   |

### 部署建议

```yaml
# 轻量部署 (cosyvoice1)
environment:
  - TTS_MODEL_MODE=cosyvoice1
  - LOG_LEVEL=WARNING
deploy:
  resources:
    limits:
      memory: 6G

# 个性化部署 (cosyvoice2)
environment:
  - TTS_MODEL_MODE=cosyvoice2
  - LOG_LEVEL=WARNING
deploy:
  resources:
    limits:
      memory: 6G

# 完整功能部署 (all)
environment:
  - TTS_MODEL_MODE=all
  - LOG_LEVEL=INFO
deploy:
  resources:
    limits:
      memory: 12G
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
| **模型下载失败** | 启动超时、网络错误 | 检查网络，重启容器：`docker-compose restart`        |
| **GPU 内存不足** | CUDA OOM 错误      | 切换 CPU 模式：设置 `DEVICE=cpu TTS_DEVICE=cpu`     |
| **端口被占用**   | 端口冲突错误       | 修改端口映射：`"8080:8000"`                         |
| **权限错误**     | 文件访问被拒绝     | 修复权限：`sudo chown -R $USER:$USER ./data ./logs` |
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

如果遇到问题无法解决：

1. **查看日志**：`docker-compose logs -f`
2. **检查配置**：确认环境变量和文件映射
3. **重启服务**：`docker-compose restart`
4. **提交问题**：访问 [项目仓库](../../issues) 提交 Issue

## 📊 部署建议

### 生产环境

```yaml
# docker-compose.prod.yml
version: "3.8"
services:
  funspeech:
    image: docker.cnb.cool/nexa/funspeech:latest
    restart: always
    environment:
      - DEBUG=false
      - LOG_LEVEL=WARNING
      - APPTOKEN=${APPTOKEN}
      - APPKEY=${APPKEY}
    volumes:
      - ./data:/app/temp
      - ./logs:/app/logs
      - ./voices:/app/voices
      - ~/.cache/modelscope:/root/.cache/modelscope
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### 开发环境

```yaml
# docker-compose.dev.yml
version: "3.8"
services:
  funspeech:
    image: docker.cnb.cool/nexa/funspeech:latest
    environment:
      - DEBUG=true
      - LOG_LEVEL=DEBUG
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/temp
      - ./logs:/app/logs
      - ./voices:/app/voices
```

---

🎉 **部署完成！**

访问 `http://localhost:8000/docs`（调试模式下）查看 API 文档，或参考 [README.md](./README.md) 了解详细使用方法。
