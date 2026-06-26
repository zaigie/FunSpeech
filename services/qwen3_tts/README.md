# Qwen3-TTS Service

Open-source local Qwen3-TTS Base Clone inference service for FunSpeech.

Default model:

```text
Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

Run locally on GPU 7:

```bash
cd services/qwen3_tts
uv sync
CUDA_VISIBLE_DEVICES=7 \
HF_ENDPOINT=https://hf-mirror.com \
PORT=8005 \
QWEN3_TTS_DEVICE=cuda:0 \
QWEN3_TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
QWEN3_TTS_VOICES_DIR=/tmp/qwen3_voices \
uv run python server.py
```

The gateway selects this backend with:

```bash
TTS_ENGINE=qwen3-tts
QWEN3_TTS_SERVICE_URLS=http://localhost:8005
```

Environment variables:

| Variable | Default | Notes |
|---|---|---|
| `QWEN3_TTS_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Only Base Clone is supported by this service integration |
| `QWEN3_TTS_DEVICE` | `cuda:0` | Runtime device inside the container/process |
| `QWEN3_TTS_DTYPE` | `bfloat16` | `bfloat16` / `float16` / `float32` |
| `QWEN3_TTS_ATTN_IMPLEMENTATION` | `sdpa` | Can be changed to `flash_attention_2` if installed |
| `QWEN3_TTS_LANGUAGE` | `Auto` | `Auto` passes `None` to qwen-tts |
| `QWEN3_TTS_VOICES_DIR` | `/app/qwen3_voices` | Stores uploaded refs, `prompts/*.pt`, and `voice_registry.json` |
| `QWEN3_TTS_X_VECTOR_ONLY_MODE` | `false` | `false` requires accurate `prompt_text`; `true` uses speaker embedding only |
| `INTERNAL_SERVICE_TOKEN` | empty | If set, all HTTP calls must include `X-Internal-Token` |

Add a Base clone voice:

```bash
curl -F name=test_clone \
  -F prompt_text='参考音频对应的准确文本' \
  -F audio=@/path/to/reference.wav \
  http://localhost:8005/voices
```

The service stores reference files, prompt tensors, and `voice_registry.json` under
`QWEN3_TTS_VOICES_DIR`.

List, synthesize, delete:

```bash
curl http://localhost:8005/voices

curl -H "Content-Type: application/json" \
  -d '{"text":"你好","voice":"test_clone"}' \
  http://localhost:8005/tts/file \
  --output speech.wav

curl -X DELETE http://localhost:8005/voices/test_clone
```

Directory scan mode is also supported: put `name.wav` and `name.txt` under
`QWEN3_TTS_VOICES_DIR`, then call:

```bash
curl -X POST http://localhost:8005/voices/refresh
```

For multi-replica deployments, write to the primary replica and call
`POST /voices/reload` on other replicas, or let the gateway HTTP engine do this
when using its `voice_manager`.
