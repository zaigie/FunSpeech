# FunSpeech 网关 — CPU only,所有模型推理在 services/* 子服务里完成

FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# 系统依赖: ffmpeg/sox/libsndfile 给 librosa/soundfile 用
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        sox \
        libsox-dev \
        libsndfile1 \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/

WORKDIR /app

# lock + pyproject 单独 COPY 提升缓存命中
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# 清掉构建工具
RUN apt-get update && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/temp /app/data /app/logs \
    && chmod +x start.py

EXPOSE 8000

CMD ["python", "start.py"]
