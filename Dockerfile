FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# 安装系统依赖（音频处理所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        sox \
        libsox-dev \
        libsndfile1 \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（多阶段拷贝官方镜像里的二进制）
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/

WORKDIR /app

# 优先 sync 依赖：把 lock + pyproject 单独拷进来，最大化缓存命中
COPY pyproject.toml uv.lock ./
# CPU 构建：使用 PyTorch CPU wheel 源 + cpu extra
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cpu --no-install-project

# 拷贝业务代码
COPY . .

# 同步项目本身（等代码就位后再跑一次以装 entry，但仍 frozen）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cpu

# 清理 build-essential（只在构建期需要）
RUN apt-get update && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/temp /app/data /app/voices /app/logs \
    && chmod +x start.py

EXPOSE 8000

CMD ["python", "start.py"]
