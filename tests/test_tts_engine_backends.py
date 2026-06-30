# -*- coding: utf-8 -*-

import pytest

from app.core.config import settings
from app.core.exceptions import DefaultServerErrorException
from app.services.tts.engine import normalize_tts_engine
from app.services.tts.http_engine import make_cosyvoice3_vllm_omni_http_engine
from app.services.tts.qwen3_http_engine import make_qwen3_tts_vllm_omni_http_engine


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cosyvoice", "cosyvoice"),
        ("qwen3-tts", "qwen3-tts"),
        ("qwen3-vllm", "qwen3-tts-vllm-omni"),
        ("qwen3_tts_vllm_omni", "qwen3-tts-vllm-omni"),
        ("cosyvoice3-vllm", "cosyvoice3-vllm-omni"),
        ("cosyvoice3_vllm_omni", "cosyvoice3-vllm-omni"),
    ],
)
def test_normalize_tts_engine_aliases(raw, expected):
    assert normalize_tts_engine(raw) == expected


def test_normalize_tts_engine_rejects_unknown():
    with pytest.raises(DefaultServerErrorException):
        normalize_tts_engine("unknown")


def test_qwen3_vllm_omni_factory_uses_own_url_setting(monkeypatch):
    monkeypatch.setattr(
        settings,
        "QWEN3_TTS_VLLM_OMNI_SERVICE_URLS",
        "http://qwen3-tts-vllm-omni-0:8006,http://qwen3-tts-vllm-omni-1:8006",
    )

    engine = make_qwen3_tts_vllm_omni_http_engine()

    assert engine._primary_url == "http://qwen3-tts-vllm-omni-0:8006"
    assert engine._urls == [
        "http://qwen3-tts-vllm-omni-0:8006",
        "http://qwen3-tts-vllm-omni-1:8006",
    ]


def test_cosyvoice3_vllm_omni_factory_uses_own_url_setting(monkeypatch):
    monkeypatch.setattr(
        settings,
        "COSYVOICE3_VLLM_OMNI_SERVICE_URLS",
        "http://cosyvoice3-vllm-omni-0:8007",
    )

    engine = make_cosyvoice3_vllm_omni_http_engine()

    assert engine._primary_url == "http://cosyvoice3-vllm-omni-0:8007"
    assert engine._urls == ["http://cosyvoice3-vllm-omni-0:8007"]
