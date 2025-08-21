# FunSpeech API 部署说明

本文档介绍如何使用 Docker 部署 FunSpeech API 服务。

## 🚀 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 下载 Docker Compose 配置
curl -sSL https://cnb.cool/nexa/FunSpeech/-/git/raw/main/docker-compose.yml -o docker-compose.yml
# 启动服务
docker-compose up -d
```

### 方式二：预构建镜像

```bash
# 运行容器
docker run -d \
  --name funspeech-api \
  -p 8000:8000 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  -v ./data:/app/temp \
  -v ./logs:/app/logs \
  -v ./voices:/app/voices \
  docker.cnb.cool/nexa/funspeech:latest
```

## 📁 目录映射

| 本地目录              | 容器目录                  | 用途         |
| --------------------- | ------------------------- | ------------ |
| `~/.cache/modelscope` | `/root/.cache/modelscope` | 模型缓存     |
| `./data`              | `/app/temp`               | 临时文件     |
| `./logs`              | `/app/logs`               | 日志文件     |
| `./voices`            | `/app/voices`             | 音色克隆文件 |

## ⚙️ 环境变量

| 变量名       | 默认值    | 描述                       |
| ------------ | --------- | -------------------------- |
| `HOST`       | `0.0.0.0` | 服务地址                   |
| `PORT`       | `8000`    | 服务端口                   |
| `DEBUG`      | `false`   | 调试模式                   |
| `LOG_LEVEL`  | `INFO`    | 日志级别                   |
| `DEVICE`     | `auto`    | ASR 设备 (auto/cpu/cuda:0) |
| `TTS_DEVICE` | `auto`    | TTS 设备 (auto/cpu/cuda:0) |
| `XLS_TOKEN`  | -         | API 鉴权 token（可选）     |
| `APPKEY`     | -         | ASR 和 TTS 接口 appkey（可选） |

## 🖥️ GPU 支持

安装 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 后，修改 `docker-compose.yml`：

```yaml
# 取消注释以下配置
environment:
  - DEVICE=cuda:0
  - TTS_DEVICE=cuda:0
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 📋 服务状态检查

```bash
# 检查容器状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 健康检查
curl http://localhost:8000/stream/v1/asr/health
curl http://localhost:8000/stream/v1/tts/health

# API 文档（Debug模式下）
# http://localhost:8000/docs
```

## 🎙️ 音色克隆使用

### 添加音色文件

```bash
# 1. 将音色文件放入映射目录
cp 张三.wav 张三.txt ./voices/

# 2. 进入容器添加音色到模型
docker exec -it funspeech-api python -m app.services.tts.clone.voice_manager --add

# 3. 查看已添加的音色
docker exec -it funspeech-api python -m app.services.tts.clone.voice_manager --list
```

### 音色文件要求

- **音频格式**：WAV 格式，建议采样率 16kHz 或以上
- **音频长度**：3-30 秒，内容清晰无杂音
- **文本内容**：与音频内容完全一致
- **文件命名**：音频和文本使用相同的文件名

### 目录结构

```
./voices/
├── 张三.wav               # 参考音频文件
├── 张三.txt               # 对应的参考文本
├── voice_registry.json   # 音色注册表（自动生成）
└── spk/                   # 模型特征文件（自动生成）
    └── spk2info.pt
```

## 🔧 本地构建

```bash
# 构建镜像
docker build -t funspeech:local .

# 运行
docker run -d \
  --name funspeech-api \
  -p 8000:8000 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  -v ./data:/app/temp \
  -v ./voices:/app/voices \
  funspeech:local
```

## 🚨 常见问题

### 模型下载失败

- 检查网络连接
- 重启容器：`docker-compose restart`

### GPU 内存不足

- 使用 CPU 模式：`DEVICE=cpu TTS_DEVICE=cpu`

### 端口冲突

- 修改端口映射：`"8080:8000"`

### 权限问题

```bash
sudo chown -R $USER:$USER ./data ./logs
```

## 🔄 更新升级

```bash
# Docker Compose
docker-compose pull
docker-compose up -d

# 手动升级
docker pull docker.cnb.cool/nexa/funspeech:latest
docker-compose up -d
```

---

🎉 部署完成后，请参考 [README.md](./README.md) 查看 API 使用文档。
