# -*- coding: utf-8 -*-

import io
import os

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.utils.audio import generate_temp_audio_path


def _wav_bytes(sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    audio = np.zeros(sample_rate // 10, dtype=np.float32)
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


class _FakeTTSEngine:
    def synthesize_speech(
        self,
        text,
        voice,
        speed,
        format,
        sample_rate,
        volume,
        prompt,
    ):
        path = generate_temp_audio_path("test_openai_tts", f".{format}")
        audio = np.zeros(sample_rate // 20, dtype=np.float32)
        sf.write(path, audio, sample_rate, format=format.upper())
        return path

    async def iter_stream_audio_chunks(self, text, voice, speed=1.0, prompt=""):
        yield np.zeros((1, 240), dtype=np.float32), 24000


class _FakeASREngine:
    def transcribe_file(self, audio_path, **kwargs):
        assert os.path.exists(audio_path)
        return "hello world"


class _FakeModelManager:
    def list_models(self):
        return [{"id": "qwen3-asr-flash"}]

    def get_asr_engine(self, model_id=None):
        assert model_id in (None, "qwen3-asr-flash")
        return _FakeASREngine()


def test_openai_speech_uses_v1_path_and_old_path_is_removed(monkeypatch, tmp_path):
    from app.api.v1 import openai as openai_routes

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APPTOKEN", None)
    monkeypatch.setattr(openai_routes, "get_tts_engine", lambda: _FakeTTSEngine())

    client = TestClient(app)

    old_response = client.post(
        "/openai/v1/audio/speech",
        json={"model": "tts-1", "input": "hello", "voice": "中文女"},
    )
    assert old_response.status_code == 404

    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "gpt-4o-mini-tts",
            "input": "hello",
            "voice": {"id": "中文女"},
            "response_format": "wav",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content.startswith(b"RIFF")


def test_openai_speech_streams_pcm(monkeypatch, tmp_path):
    from app.api.v1 import openai as openai_routes

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APPTOKEN", None)
    monkeypatch.setattr(openai_routes, "get_tts_engine", lambda: _FakeTTSEngine())

    client = TestClient(app)
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "hello",
            "voice": "中文女",
            "response_format": "pcm",
            "stream_format": "audio",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert len(response.content) == 480


def test_openai_transcriptions_json(monkeypatch, tmp_path):
    from app.api.v1 import openai as openai_routes

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APPTOKEN", None)
    monkeypatch.setattr(openai_routes, "get_model_manager", lambda: _FakeModelManager())

    client = TestClient(app)
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "json"},
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}


def test_openai_transcriptions_stream(monkeypatch, tmp_path):
    from app.api.v1 import openai as openai_routes

    monkeypatch.setattr(settings, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APPTOKEN", None)
    monkeypatch.setattr(openai_routes, "get_model_manager", lambda: _FakeModelManager())

    client = TestClient(app)
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "qwen3-asr-flash", "stream": "true"},
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "transcript.text.delta" in response.text
    assert "transcript.text.done" in response.text
