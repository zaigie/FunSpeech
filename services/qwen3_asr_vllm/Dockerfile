# qwen3-asr-vllm 子服务 — Qwen3-ASR + 进程内 vLLM 加速
#
# vLLM 自带 CUDA wheel, 基础镜像选 Python + CUDA runtime 即可。
# Layer 顺序: 基础镜像 → 系统依赖 → uv 二进制 → uv sync 依赖 → COPY 代码 + uv sync 项目。
# 需要 BuildKit (Docker 23+ 默认开启, 或 DOCKER_BUILDKIT=1)。

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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
    UV_PYTHON_DOWNLOADS=automatic \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        ffmpeg libsndfile1 curl ca-certificates git \
        gcc g++

# uv 装到 apt 装的 python3.10 (本 base 是裸 CUDA 镜像, 不自带 python)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3.10 -m pip install --no-cache-dir uv==0.11.8

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" no_proxy="" NO_PROXY=""

EXPOSE 8003

# 直接走 venv 里的 python, 跳过 `uv run` 的 bytecode 校验
CMD ["/app/.venv/bin/python", "server.py"]
