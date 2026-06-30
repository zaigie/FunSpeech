# -*- coding: utf-8 -*-
"""CosyVoice3 vLLM-Omni facade service."""

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


MODEL_ID = first_env(
    ["COSYVOICE3_OMNI_MODEL_ID", "COSYVOICE3_MODEL_ID"],
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
)

config = OmniTTSConfig(
    service_name="cosyvoice3-vllm-omni",
    public_title="funspeech-cosyvoice3-vllm-omni-service",
    model_id=MODEL_ID,
    task_type="clone",
    default_voice=first_env(["COSYVOICE3_OMNI_DEFAULT_VOICE"], ""),
    default_language=first_env(["COSYVOICE3_OMNI_LANGUAGE"], ""),
    sample_rate=int(first_env(["COSYVOICE3_OMNI_SAMPLE_RATE"], "24000")),
    supports_presets=False,
    supports_voice_clone=True,
    include_task_type=False,
    prefer_uploaded_voice=False,
    require_ref_for_voice=bool_env("COSYVOICE3_OMNI_FORCE_REF_PER_REQUEST", True),
    deploy_config_name=first_env(["COSYVOICE3_OMNI_DEPLOY_CONFIG"], "cosyvoice3.yaml"),
    voices_dir=Path(first_env(["COSYVOICE3_OMNI_VOICES_DIR", "VOICES_DIR"], "/app/cosyvoice3_omni_voices")),
    port=int(first_env(["PORT"], "8007")),
    internal_token=first_env(["INTERNAL_SERVICE_TOKEN"], ""),
    api_base=first_env(["COSYVOICE3_OMNI_API_BASE", "VLLM_OMNI_API_BASE"], "http://127.0.0.1:8091"),
    api_key=first_env(["COSYVOICE3_OMNI_API_KEY", "VLLM_OMNI_API_KEY"], "EMPTY"),
    start_omni=bool_env("COSYVOICE3_OMNI_START_SERVER", bool_env("VLLM_OMNI_START_SERVER", True)),
    omni_host=first_env(["COSYVOICE3_OMNI_HOST", "VLLM_OMNI_HOST"], "127.0.0.1"),
    omni_port=int(first_env(["COSYVOICE3_OMNI_PORT", "VLLM_OMNI_PORT"], "8091")),
    startup_timeout_sec=float(first_env(["COSYVOICE3_OMNI_STARTUP_TIMEOUT"], "900")),
    request_timeout_sec=float(first_env(["COSYVOICE3_OMNI_REQUEST_TIMEOUT"], "300")),
    serve_command=first_env(["COSYVOICE3_OMNI_SERVE_COMMAND", "VLLM_OMNI_SERVE_COMMAND"], "vllm"),
    gpu_memory_utilization=first_env(["COSYVOICE3_OMNI_GPU_MEM", "VLLM_OMNI_GPU_MEM"], ""),
    stage_overrides=first_env(["COSYVOICE3_OMNI_STAGE_OVERRIDES", "VLLM_OMNI_STAGE_OVERRIDES"], ""),
    extra_serve_args=first_env(["COSYVOICE3_OMNI_EXTRA_ARGS", "VLLM_OMNI_EXTRA_ARGS"], ""),
    trust_remote_code=bool_env("COSYVOICE3_OMNI_TRUST_REMOTE_CODE", True),
    enforce_eager=bool_env("COSYVOICE3_OMNI_ENFORCE_EAGER", False),
    no_async_chunk=bool_env("COSYVOICE3_OMNI_NO_ASYNC_CHUNK", False),
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
