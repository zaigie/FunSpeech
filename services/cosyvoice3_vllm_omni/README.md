# CosyVoice3 vLLM-Omni Service

FunSpeech-compatible facade for CosyVoice3 served by vLLM-Omni.

Default model:

```text
FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

This service exposes the same `/tts/file`, `/tts/stream`, and `/voices` protocol
as the legacy `services/cosyvoice` container, while internally using
`vllm serve ... --omni`.

CosyVoice3 has no built-in preset voices in vLLM-Omni Speech API mode, so upload
a clone voice before synthesis:

```bash
curl -F name=benchclone \
  -F prompt_text='参考音频对应的准确文本' \
  -F audio=@/path/to/reference.wav \
  http://localhost:8007/voices
```

Then synthesize:

```bash
curl -H "Content-Type: application/json" \
  -d '{"text":"你好，这是 vLLM-Omni CosyVoice3 测试。","voice":"benchclone"}' \
  http://localhost:8007/tts/file \
  --output speech.wav
```

Key environment variables:

| Variable | Default | Notes |
|---|---|---|
| `COSYVOICE3_OMNI_MODEL_ID` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Model id or local path |
| `COSYVOICE3_OMNI_VOICES_DIR` | `/app/cosyvoice3_omni_voices` | Uploaded references + registry |
| `COSYVOICE3_OMNI_PORT` | `8091` | Internal vLLM-Omni port |
| `PORT` | `8007` | Public facade port |
| `COSYVOICE3_OMNI_SERVE_COMMAND` | `vllm` | Override only if your image exposes a different CLI |
| `COSYVOICE3_OMNI_STAGE_OVERRIDES` | empty | Passed to `vllm serve --stage-overrides` |
| `COSYVOICE3_OMNI_FORCE_REF_PER_REQUEST` | `true` | Sends stored `ref_audio/ref_text` per request |

Optional GPU ONNX Runtime build:

```bash
docker build \
  --build-arg INSTALL_ONNXRUNTIME_GPU=true \
  -f services/cosyvoice3_vllm_omni/Dockerfile \
  -t funspeech/cosyvoice3-vllm-omni:gpu-ort .

COSYVOICE3_OMNI_INSTALL_ONNXRUNTIME_GPU=true \
  docker compose build cosyvoice3-vllm-omni-0
```

4090 24G benchmark: GPU ORT improved single-request standard RTF from about
0.57 to 0.51, but peak throughput did not improve and VRAM increased from
about 10.8 GiB to 15.3 GiB net. The default image therefore keeps CPU ORT.
