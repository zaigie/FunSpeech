# -*- coding: utf-8 -*-
"""dolphin 子服务 smoke test"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "test-token")
    import server  # type: ignore[import-not-found]

    return TestClient(server.app, raise_server_exceptions=False)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"


def test_asr_file_requires_token(client):
    files = {"audio": ("a.wav", b"x", "audio/wav")}
    r = client.post("/asr/file", files=files)
    assert r.status_code == 401


def test_asr_file_rejects_bad_token(client):
    files = {"audio": ("a.wav", b"x", "audio/wav")}
    r = client.post(
        "/asr/file", files=files, headers={"X-Internal-Token": "wrong"}
    )
    assert r.status_code == 401
