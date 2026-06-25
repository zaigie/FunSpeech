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

Add a Base clone voice:

```bash
curl -F name=test_clone \
  -F prompt_text='参考音频对应的准确文本' \
  -F audio=@/path/to/reference.wav \
  http://localhost:8005/voices
```

The service stores reference files, prompt tensors, and `voice_registry.json` under
`QWEN3_TTS_VOICES_DIR`.
