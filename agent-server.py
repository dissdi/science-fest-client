from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List
import os
import time
import uuid
import threading

import cv2
import requests


# ----------------------------
# App
# ----------------------------
app = FastAPI()


# ----------------------------
# Config (env)
# ----------------------------
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

DEMO_AUTO_TOOLS = os.getenv("DEMO_AUTO_TOOLS", "true").lower() == "true"
DASHBOARD_UPLOAD_URL = os.getenv("DASHBOARD_UPLOAD_URL", "").strip()

ALLOWED_UPLOAD_HOSTS = {
    host.strip()
    for host in os.getenv("ALLOWED_UPLOAD_HOSTS", "").split(",")
    if host.strip()
}
# 예: ALLOWED_UPLOAD_HOSTS="127.0.0.1,localhost,192.168.0.10"

CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
CAM_W = int(os.getenv("CAM_W", "1280"))
CAM_H = int(os.getenv("CAM_H", "720"))
CAM_WARMUP = int(os.getenv("CAM_WARMUP", "8"))

# Windows면 CAP_DSHOW가 안정적인 편, 그 외 OS면 기본값 사용
CAM_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else 0


# ----------------------------
# In-memory storage (demo)
# ----------------------------
SESSIONS = set()
FILES: Dict[str, str] = {}  # file_id -> saved_path


# ----------------------------
# Locks
# ----------------------------
CAM_LOCK = threading.Lock()


# ----------------------------
# Schemas
# ----------------------------
class ConsentIn(BaseModel):
    consent: bool


class ChatSendIn(BaseModel):
    session_id: str
    message: str
    file_ids: List[str] = []


class ResetIn(BaseModel):
    session_id: str


# ----------------------------
# Helpers
# ----------------------------
def extract_text_from_files(paths: List[str], max_chars: int = 12000) -> str:
    """데모용: txt/md/log 위주. 지원 안 하는 형식은 파일명만 컨텍스트로."""
    chunks: List[str] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in [".txt", ".md", ".log"]:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    chunks.append(f"\n\n[FILE:{os.path.basename(p)}]\n" + f.read())
            except Exception:
                continue
        else:
            chunks.append(f"\n\n[FILE:{os.path.basename(p)}] (text extract skipped)")
    return ("".join(chunks))[:max_chars]


def call_llm(message: str, context_text: str) -> Dict[str, Any]:
    """
    TODO: 실제 외부 LLM API 호출로 교체.
    반환 형식:
      { "assistant_text": "...", "tool_request": [ {name, args}, ... ] }
    """
    tool_request = []
    if "요약" in message and "capture" in (message + context_text).lower():
        tool_request = [
            {"name": "camera_capture", "args": {"reason": "demo"}},
            {"name": "send_network_request", "args": {"kind": "image"}},
        ]
    return {
        "assistant_text": f"(LLM 데모) '{message}' 처리했어요.",
        "tool_request": tool_request,
    }


def camera_capture_to_file(out_dir: str = STORAGE_DIR) -> str:
    """OpenCV로 현재 프레임 1장 저장."""
    with CAM_LOCK:
        filename = f"capture_{int(time.time())}.jpg"
        path = os.path.join(out_dir, filename)

        cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"camera_open_failed (index={CAM_INDEX})")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

        ok = False
        frame = None
        for _ in range(CAM_WARMUP):
            ok, frame = cap.read()

        if not ok or frame is None:
            cap.release()
            raise RuntimeError("camera_read_failed")

        ok2 = cv2.imwrite(path, frame)
        cap.release()

        if not ok2 or not os.path.exists(path):
            raise RuntimeError("image_save_failed")

        return path


def is_allowed_upload(url: str) -> bool:
    if not url:
        return False
    try:
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
        if not ALLOWED_UPLOAD_HOSTS:
            return False
        return host in ALLOWED_UPLOAD_HOSTS
    except Exception:
        return False


def upload_image_to_dashboard(image_path: str, description: str = "") -> Dict[str, Any]:
    if not DASHBOARD_UPLOAD_URL:
        raise RuntimeError("DASHBOARD_UPLOAD_URL is not set")
    if not is_allowed_upload(DASHBOARD_UPLOAD_URL):
        raise RuntimeError("upload host not allowed")

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        data = {"description": description}
        r = requests.post(DASHBOARD_UPLOAD_URL, files=files, data=data, timeout=5)
        r.raise_for_status()
        return {"status_code": r.status_code}


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "version": "0.1.0"}


@app.post("/consent")
def consent(data: ConsentIn):
    if not data.consent:
        raise HTTPException(status_code=400, detail="consent must be true")
    session_id = uuid.uuid4().hex[:8]
    SESSIONS.add(session_id)
    return {"session_id": session_id}


@app.post("/files")
async def upload_file(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex[:8]
    save_path = os.path.join(STORAGE_DIR, f"{file_id}_{file.filename}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    FILES[file_id] = save_path
    return {
        "file_id": file_id,
        "filename": file.filename,
        "mime": file.content_type,
        "size": len(content),
    }


@app.post("/chat/send")
def chat_send(data: ChatSendIn):
    if data.session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")

    events = [{"type": "analysis_started"}]

    # 첨부 파일 경로 확인
    attached_paths: List[str] = []
    for fid in data.file_ids:
        if fid not in FILES:
            raise HTTPException(status_code=400, detail=f"unknown_file_id: {fid}")
        attached_paths.append(FILES[fid])

    # 1) 파일 텍스트 추출
    context_text = extract_text_from_files(attached_paths)
    events.append({"type": "file_text_extracted", "detail": {"n_files": len(attached_paths)}})

    # 2) LLM 호출
    llm_out = call_llm(data.message, context_text)
    events.append({"type": "llm_called"})

    assistant_text = llm_out.get("assistant_text", "")
    tool_request = llm_out.get("tool_request", [])
    if tool_request:
        events.append({"type": "tool_planned", "detail": {"tools": [t.get("name") for t in tool_request]}})

    # 3) 도구 자동 실행(부스 모드)
    if DEMO_AUTO_TOOLS and tool_request:
        try:
            if any(t.get("name") == "camera_capture" for t in tool_request):
                img_path = camera_capture_to_file()
                events.append({"type": "camera_captured", "detail": {"path": os.path.basename(img_path)}})

                if any(t.get("name") == "send_network_request" for t in tool_request):
                    up = upload_image_to_dashboard(img_path, description="demo auto-run")
                    events.append({"type": "uploaded", "detail": up})

        except Exception as e:
            events.append({"type": "error", "detail": {"message": str(e)}})

    events.append({"type": "done"})
    return {"assistant": assistant_text, "events": events}


@app.post("/chat/reset")
def chat_reset(data: ResetIn):
    if data.session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")
    return {"ok": True}
