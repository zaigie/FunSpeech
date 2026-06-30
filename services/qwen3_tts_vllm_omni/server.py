# -*- coding: utf-8 -*-
"""Qwen3-TTS vLLM-Omni facade service."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from services.vllm_omni_tts_common.facade import (
    OmniTTSConfig,
    OmniTTSFacade,
    bool_env,
    first_env,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _task_type_from_model(model_id: str) -> str:
    lowered = model_id.lower()
    if "voicedesign" in lowered:
        return "VoiceDesign"
    if "customvoice" in lowered:
        return "CustomVoice"
    return "Base"


MODEL_ID = first_env(
    ["QWEN3_TTS_OMNI_MODEL_ID", "QWEN3_TTS_MODEL_ID"],
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
)
TASK_TYPE = first_env(["QWEN3_TTS_OMNI_TASK_TYPE"], _task_type_from_model(MODEL_ID))

config = OmniTTSConfig(
    service_name="qwen3-tts-vllm-omni",
    public_title="funspeech-qwen3-tts-vllm-omni-service",
    model_id=MODEL_ID,
    task_type=TASK_TYPE,
    default_voice=first_env(["QWEN3_TTS_OMNI_DEFAULT_VOICE"], "vivian"),
    default_language=first_env(["QWEN3_TTS_OMNI_LANGUAGE", "QWEN3_TTS_LANGUAGE"], "Auto"),
    sample_rate=int(first_env(["QWEN3_TTS_OMNI_SAMPLE_RATE"], "24000")),
    supports_presets=TASK_TYPE.lower() in ("customvoice", "voicedesign"),
    supports_voice_clone=TASK_TYPE.lower() == "base",
    include_task_type=True,
    prefer_uploaded_voice=True,
    require_ref_for_voice=bool_env("QWEN3_TTS_OMNI_FORCE_REF_PER_REQUEST", False),
    deploy_config_name=first_env(["QWEN3_TTS_OMNI_DEPLOY_CONFIG"], "qwen3_tts.yaml"),
    voices_dir=Path(
        first_env(
            ["QWEN3_TTS_OMNI_VOICES_DIR", "QWEN3_TTS_VOICES_DIR", "VOICES_DIR"],
            "/app/qwen3_omni_voices",
        )
    ),
    port=int(first_env(["PORT"], "8006")),
    internal_token=first_env(["INTERNAL_SERVICE_TOKEN"], ""),
    api_base=first_env(["QWEN3_TTS_OMNI_API_BASE", "VLLM_OMNI_API_BASE"], "http://127.0.0.1:8091"),
    api_key=first_env(["QWEN3_TTS_OMNI_API_KEY", "VLLM_OMNI_API_KEY"], "EMPTY"),
    start_omni=bool_env("QWEN3_TTS_OMNI_START_SERVER", bool_env("VLLM_OMNI_START_SERVER", True)),
    omni_host=first_env(["QWEN3_TTS_OMNI_HOST", "VLLM_OMNI_HOST"], "127.0.0.1"),
    omni_port=int(first_env(["QWEN3_TTS_OMNI_PORT", "VLLM_OMNI_PORT"], "8091")),
    startup_timeout_sec=float(first_env(["QWEN3_TTS_OMNI_STARTUP_TIMEOUT"], "900")),
    request_timeout_sec=float(first_env(["QWEN3_TTS_OMNI_REQUEST_TIMEOUT"], "300")),
    serve_command=first_env(["QWEN3_TTS_OMNI_SERVE_COMMAND", "VLLM_OMNI_SERVE_COMMAND"], "vllm"),
    gpu_memory_utilization=first_env(["QWEN3_TTS_OMNI_GPU_MEM", "VLLM_OMNI_GPU_MEM"], ""),
    stage_overrides=first_env(["QWEN3_TTS_OMNI_STAGE_OVERRIDES", "VLLM_OMNI_STAGE_OVERRIDES"], ""),
    extra_serve_args=first_env(["QWEN3_TTS_OMNI_EXTRA_ARGS", "VLLM_OMNI_EXTRA_ARGS"], ""),
    trust_remote_code=bool_env("QWEN3_TTS_OMNI_TRUST_REMOTE_CODE", True),
    enforce_eager=bool_env("QWEN3_TTS_OMNI_ENFORCE_EAGER", False),
    no_async_chunk=bool_env("QWEN3_TTS_OMNI_NO_ASYNC_CHUNK", False),
    env={
        "HF_ENDPOINT": os.getenv("HF_ENDPOINT", ""),
        "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE", ""),
        "TRANSFORMERS_OFFLINE": os.getenv("TRANSFORMERS_OFFLINE", ""),
    },
)

facade = OmniTTSFacade(config)
app = facade.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.port)
