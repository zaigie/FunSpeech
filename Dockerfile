# FunSpeech 网关 — CPU only,所有模型推理在 services/* 子服务里完成
#
# Layer 顺序按"变化频率从低到高"排列, 让本地重 build 在改代码时
# 只重跑最后两层:
#   1. base image
#   2. 系统依赖 (apt)            — 极少变
#   3. uv 二进制                — 极少变
#   4. 运行期目录 (mkdir)       — 与代码无关
#   5. COPY pyproject.toml uv.lock + uv sync 依赖
#   6. COPY . . + uv sync 项目本身
#
# 启用 BuildKit (Docker 23+ 默认开)。低版本:
#   export DOCKER_BUILDKIT=1
#
# 用法 (本机有 http://127.0.0.1:7890 代理时):
#   docker build --build-arg HTTP_PROXY=http://host.docker.internal:7890 \
#                --build-arg HTTPS_PROXY=http://host.docker.internal:7890 -t funspeech/gateway .

FROM python:3.10-slim AS runtime

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
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# ---------- 系统依赖 (一次装完, 不再 install→remove 来回折腾) ----------
# build-essential 留下: wetext / 老 sdist 在 wheel rebuild 时仍可能需要,
# 卸载省的几十 MB 不值得多一层 + 多一次 apt-get update 的时间。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        sox \
        libsox-dev \
        libsndfile1 \
        build-essential \
        curl \
        ca-certificates

# ---------- uv (装到 base 自带 python, 不走 ghcr) ----------
# pip 走构建期 HTTP_PROXY, 不依赖 ghcr.io/astral-sh/uv 镜像。
# uv 仅用于 sync 项目依赖到 /app/.venv, 自身在哪个 python 不关键。
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir uv==0.11.8

WORKDIR /app

# ---------- 运行期目录 (与代码无关, 单独成层) ----------
RUN mkdir -p /app/temp /app/data /app/logs

# ---------- Python 依赖层 (仅看 pyproject + lock, 改代码不会失效) ----------
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# ---------- 代码层 (变化最频繁, 放最后) ----------
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
    && chmod +x start.py

# 清掉运行期代理 ENV
ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" no_proxy="" NO_PROXY=""

EXPOSE 8000

CMD ["python", "start.py"]
