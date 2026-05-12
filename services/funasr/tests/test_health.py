# -*- coding: utf-8 -*-
"""funasr 子服务 smoke test

只覆盖不需模型权重的路径(导入/健康检查/鉴权)。
完整的 transcribe 测试见 tests/(主项目 e2e)。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# 让 import server 能找到 sibling
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client(monkeypatch):
    # 防止 lifespan 真去加载模型
    monkeypatch.setenv("ASR_MODEL_MODE", "offline")
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "test-token")

    import server  # type: ignore[import-not-found]

    # lifespan 加载模型可能很慢甚至失败,这里直接走没有 lifespan 的 client
    return TestClient(server.app, raise_server_exceptions=False)


def test_health_no_token_required(client):
    # lifespan 被绕过, 模型未加载 → /health 应返回 503 status="starting"
    # 表示服务还没就绪 (但端点是公开的, 不需 token)
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "starting"
    assert "device" in body


def test_asr_file_requires_token(client):
    # 不带 token 应该 401
    files = {"audio": ("a.wav", b"RIFFsomething", "audio/wav")}
    r = client.post("/asr/file", files=files)
    assert r.status_code == 401


def test_asr_file_rejects_bad_token(client):
    files = {"audio": ("a.wav", b"RIFFsomething", "audio/wav")}
    r = client.post(
        "/asr/file",
        files=files,
        headers={"X-Internal-Token": "wrong"},
    )
    assert r.status_code == 401
