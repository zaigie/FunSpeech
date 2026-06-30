# cosyvoice3-vllm-omni 子服务 — CosyVoice3 via vLLM-Omni

FROM vllm/vllm-openai:v0.22.1

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
    PYTHONPATH=/app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg sox libsndfile1 curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
      "vllm-omni==0.22.0" \
      "fastapi==0.115.6" \
      "uvicorn[standard]==0.30.0" \
      "python-multipart==0.0.6" \
      "httpx>=0.28.1" \
      "soundfile>=0.12.1" \
      "websockets>=11.0"

ARG INSTALL_ONNXRUNTIME_GPU=false

# 可选: vllm-omni 依赖的 onnxruntime wheel 只有 CPU provider。
# CosyVoice3 的部分后处理走 ONNX Runtime; GPU wheel 会提高单路延迟, 但实测显存更高且峰值吞吐不一定更好。
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ "$INSTALL_ONNXRUNTIME_GPU" = "true" ]; then \
      pip install --no-cache-dir "onnxruntime-gpu==1.27.0"; \
    fi

WORKDIR /app

COPY services/vllm_omni_tts_common /app/services/vllm_omni_tts_common
COPY services/cosyvoice3_vllm_omni /app/services/cosyvoice3_vllm_omni

ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" no_proxy="" NO_PROXY=""

EXPOSE 8007

ENTRYPOINT []
CMD ["python3", "-m", "services.cosyvoice3_vllm_omni.server"]
