FROM python:3.10-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# 系统换源(debian bookworm)
RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
        sed -i 's/security.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list; \
    fi && \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
        sed -i 's/security.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources; \
    fi

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    libsox-dev \
    sox \
    ffmpeg \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

COPY requirements.txt /tmp/app_requirements.txt

# 设置pip源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 先复制项目文件
COPY . .

# 替换app/services/tts/third_party/CosyVoice/requirements.txt中的pip源
RUN sed -i 's|https://download.pytorch.org/whl/cu121|https://mirrors.aliyun.com/pytorch-wheels|g' /app/app/services/tts/third_party/CosyVoice/requirements.txt

# 安装依赖
RUN pip install -r /app/app/services/tts/third_party/CosyVoice/requirements.txt
RUN pip install -r /tmp/app_requirements.txt

# 清理缓存
RUN pip cache purge

# 创建必要的目录
RUN mkdir -p /app/temp /app/voices /app/logs

# 设置权限
RUN chmod +x start.py

# 暴露端口
EXPOSE 8000

# 设置默认命令
CMD ["python", "start.py"] 