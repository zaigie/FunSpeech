# cosyvoice 子服务 — CosyVoice2 / CosyVoice3 + 进程内 vLLM (可选)
#
# Layer 顺序:
#   1. 基础镜像
#   2. 系统依赖 (apt + cache mount)
#   3. uv 二进制
#   4. COPY pyproject + lock + uv sync 依赖    (依赖变化时失效)
#   5. COPY third_party (上游 CosyVoice 源码)  (submodule 升级才失效, 大量代码)
#   6. COPY 其它代码 + uv sync 项目             (改 server.py 走这一步)
#
# third_party 单独成层是关键优化: 上游 ~200MB+ 代码, 与本服务 server.py
# 改动频率完全不同, 拆开后日常迭代 build 极快。
#
# 需要 BuildKit (Docker 23+ 默认开启, 或 DOCKER_BUILDKIT=1)。
#
# **注意**: build 之前必须在宿主机执行
#   git submodule update --init --recursive
# 否则 third_party/CosyVoice 是空目录, 镜像启动 ModuleNotFoundError。

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
        ffmpeg sox libsox-dev libsndfile1 build-essential \
        curl ca-certificates git

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/

WORKDIR /app

# Python 依赖层
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# 上游 CosyVoice 源码层 (改 server.py 不让它失效)
COPY third_party ./third_party

# 项目代码层 (server.py + tests + README 等小变化频繁文件)
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" no_proxy="" NO_PROXY=""

EXPOSE 8004

CMD ["uv", "run", "python", "server.py"]
