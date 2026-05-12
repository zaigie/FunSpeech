# -*- coding: utf-8 -*-
"""cosyvoice 子服务 smoke test"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "test-token")
    monkeypatch.setenv("VOICES_DIR", str(tmp_path / "voices"))

    import server  # type: ignore[import-not-found]

    return TestClient(server.app, raise_server_exceptions=False)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "starting"


def test_tts_file_requires_token(client):
    r = client.post("/tts/file", json={"text": "hi"})
    assert r.status_code == 401


def test_tts_file_rejects_bad_token(client):
    r = client.post(
        "/tts/file",
        json={"text": "hi"},
        headers={"X-Internal-Token": "wrong"},
    )
    assert r.status_code == 401


def test_voices_list_empty(client):
    r = client.get("/voices", headers={"X-Internal-Token": "test-token"})
    assert r.status_code == 200
    body = r.json()
    assert "preset" in body
    assert "clone" in body


def test_text_normalize_no_engine(client):
    # 没加载任何模型时, 应回退到 [text] 而不是报错
    r = client.post(
        "/text/normalize",
        json={"text": "hello world"},
        headers={"X-Internal-Token": "test-token"},
    )
    assert r.status_code == 200
    assert r.json()["sentences"] == ["hello world"]
