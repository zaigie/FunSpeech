# Qwen3-TTS vLLM-Omni Service

FunSpeech-compatible facade for Qwen3-TTS served by vLLM-Omni.

Default model:

```text
Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

The public service keeps the same protocol as `services/qwen3_tts`:

- `GET /health`
- `POST /tts/file`
- `WS /tts/stream`
- `GET/POST/DELETE /voices`

Internally it starts `vllm serve ... --omni` on `127.0.0.1:8091` and calls
the OpenAI-compatible `/v1/audio/speech` and `/v1/audio/voices` endpoints.

4090 24G benchmark: Base Clone reached 2.41 req/s peak and 2.06 req/s at
N=8, with N=8 mean standard RTF about 0.81. This is much faster than the
legacy local `qwen-tts` backend, but uses about 19-20 GiB net VRAM.

Key environment variables:

| Variable | Default | Notes |
|---|---|---|
| `QWEN3_TTS_OMNI_MODEL_ID` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Can be Base, CustomVoice, or VoiceDesign |
| `QWEN3_TTS_OMNI_TASK_TYPE` | inferred from model | `Base` / `CustomVoice` / `VoiceDesign` |
| `QWEN3_TTS_OMNI_VOICES_DIR` | `/app/qwen3_omni_voices` | Uploaded references + registry |
| `QWEN3_TTS_OMNI_PORT` | `8091` | Internal vLLM-Omni port |
| `PORT` | `8006` | Public facade port |
| `QWEN3_TTS_OMNI_SERVE_COMMAND` | `vllm` | Override only if your image exposes a different CLI |
| `QWEN3_TTS_OMNI_STAGE_OVERRIDES` | empty | Passed to `vllm serve --stage-overrides` |
| `QWEN3_TTS_OMNI_START_SERVER` | `true` | Set false to proxy an external vLLM-Omni server |

For Base clone, upload a voice first:

```bash
curl -F name=benchclone \
  -F prompt_text='参考音频对应的准确文本' \
  -F audio=@/path/to/reference.wav \
  http://localhost:8006/voices
```

Then synthesize:

```bash
curl -H "Content-Type: application/json" \
  -d '{"text":"你好，这是 vLLM-Omni Qwen3-TTS 测试。","voice":"benchclone"}' \
  http://localhost:8006/tts/file \
  --output speech.wav
```
