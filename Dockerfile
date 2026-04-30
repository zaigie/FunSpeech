# FunSpeech 网关 — CPU only,所有模型推理在 services/* 子服务里完成

FROM python:3.10-slim AS runtime

# ---------- 构建期 HTTP 代理 ----------
# 用法: docker build --build-arg HTTP_PROXY=http://host.docker.internal:7890 ...
# 或在 docker-compose.yml 的 build.args 里集中配
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="localhost,127.0.0.1,*.local"
ENV http_proxy=$HTTP_PROXY \
    https_proxy=$HTTPS_PROXY \
    HTTP_PROXY=$HTTP_PROXY \
    HTTPS_PROXY=$HTTPS_PROXY \
    no_proxy=$NO_PROXY \
    NO_PROXY=$NO_PROXY

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# 系统依赖
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

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN apt-get update && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/temp /app/data /app/logs \
    && chmod +x start.py

# 清掉运行期代理 ENV
ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" no_proxy="" NO_PROXY=""

EXPOSE 8000

CMD ["python", "start.py"]
