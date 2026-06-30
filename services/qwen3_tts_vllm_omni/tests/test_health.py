from fastapi.testclient import TestClient

from services.qwen3_tts_vllm_omni import server


def test_health_starting_when_omni_not_started(monkeypatch):
    monkeypatch.setattr(server.facade, "_load_failed", False)
    monkeypatch.setattr(server.facade, "_ready", False)
    monkeypatch.setattr(server.facade, "_check_omni_ready", lambda timeout=1.5: False)

    client = TestClient(server.app)
    resp = client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "starting"
    assert body["backend"] == "vllm-omni"


def test_default_serve_command_is_vllm():
    cmd = server.facade._build_omni_command()

    assert cmd[:2] == ["vllm", "serve"]
    assert "--omni" in cmd
