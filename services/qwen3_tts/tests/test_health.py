from fastapi.testclient import TestClient

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server


def test_health_unhealthy_when_load_failed(monkeypatch):
    monkeypatch.setattr(server, "_load_failed", True)
    monkeypatch.setattr(server, "_load_error_msg", "boom")
    monkeypatch.setattr(server, "_model", None)

    client = TestClient(server.app)
    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["status"] == "unhealthy"
