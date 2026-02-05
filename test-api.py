# test_api.py
import os
import io
import pytest
from fastapi.testclient import TestClient

# ✅ 여기를 너의 FastAPI 파일명에 맞춰 바꿔줘
# 예: server.py면 -> from server import app
from agent_server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _set_env_and_disable_tools(monkeypatch):
    """
    테스트 중엔 카메라/업로드 같은 실제 동작이 절대 실행되지 않도록 막기.
    """
    monkeypatch.setenv("DEMO_AUTO_TOOLS", "false")  # /chat/send에서 tool auto-run 비활성
    # 혹시 모듈 로딩 시 이미 읽혔더라도, 아래 테스트들이 tool 실행을 요구하지 않게 구성함.


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "version" in data


def test_consent_required_blocks_chat():
    # consent 없이 chat/send 호출하면 403
    r = client.post("/chat/send", json={"session_id": "nosession", "message": "hi", "file_ids": []})
    assert r.status_code == 403
    assert r.json()["detail"] == "consent_required"


def test_consent_returns_session_id():
    r = client.post("/consent", json={"consent": True})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert isinstance(data["session_id"], str)
    assert len(data["session_id"]) > 0


def test_upload_file_and_chat_send_with_file():
    # 1) consent
    r = client.post("/consent", json={"consent": True})
    session_id = r.json()["session_id"]

    # 2) file upload (txt)
    fake_txt = b"Hello file context!\nThis is a test."
    files = {
        "file": ("demo.txt", io.BytesIO(fake_txt), "text/plain")
    }
    r2 = client.post("/files", files=files)
    assert r2.status_code == 200
    file_info = r2.json()
    assert "file_id" in file_info
    file_id = file_info["file_id"]

    # 3) chat/send
    payload = {
        "session_id": session_id,
        "message": "이 문서 요약해줘",
        "file_ids": [file_id],
    }
    r3 = client.post("/chat/send", json=payload)
    assert r3.status_code == 200
    out = r3.json()

    assert "assistant" in out
    assert isinstance(out["assistant"], str)

    assert "events" in out
    assert isinstance(out["events"], list)

    # 이벤트 최소 흐름 확인
    types = [e.get("type") for e in out["events"]]
    assert "analysis_started" in types
    assert "file_text_extracted" in types
    assert "llm_called" in types
    assert "done" in types


def test_chat_send_unknown_file_id():
    # consent
    r = client.post("/consent", json={"consent": True})
    session_id = r.json()["session_id"]

    # unknown file_id 보내면 400
    payload = {
        "session_id": session_id,
        "message": "테스트",
        "file_ids": ["no_such_file"],
    }
    r2 = client.post("/chat/send", json=payload)
    assert r2.status_code == 400
    assert "unknown_file_id" in r2.json()["detail"]


def test_chat_reset():
    r = client.post("/consent", json={"consent": True})
    session_id = r.json()["session_id"]

    r2 = client.post("/chat/reset", json={"session_id": session_id})
    assert r2.status_code == 200
    assert r2.json()["ok"] is True
