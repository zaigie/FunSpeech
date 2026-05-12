# dolphin 子服务 — DataoceanAI Dolphin (仅离线)
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

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl ca-certificates \
        gcc g++  # espnet/ctc-segmentation 需要编译, 见 2701777

# uv 装到 base 自带的 python (pytorch/pytorch:*-runtime 自带 python + pip)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir uv==0.11.8

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" no_proxy="" NO_PROXY=""

EXPOSE 8002

# 直接走 venv 里的 python, 跳过 `uv run` 的 bytecode 校验
CMD ["/app/.venv/bin/python", "server.py"]
