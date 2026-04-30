# funasr 子服务 — Paraformer / SenseVoice + VAD + PUNC
#
# Layer 顺序: 基础镜像 → 系统依赖 → uv 二进制 → uv sync 依赖 → COPY 代码 + uv sync 项目。
# 需要 BuildKit (Docker 23+ 默认开启, 或 DOCKER_BUILDKIT=1)。

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# ---------- 构建期 HTTP 代理 ----------
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
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# 系统依赖 — apt 缓存挂载, 重 build 不重下 deb
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg sox libsox-dev libsndfile1 curl ca-certificates

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/

WORKDIR /app

# Python 依赖层 — 仅看 pyproject + lock, 改 server.py 不会失效
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# 代码层
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" no_proxy="" NO_PROXY=""

EXPOSE 8001

CMD ["uv", "run", "python", "server.py"]
