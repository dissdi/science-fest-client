from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Any, Dict, List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import os
import time
import uuid
import threading
import json

import cv2
import requests
from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------
# App
# ----------------------------
load_dotenv()  # 프로젝트 루트의 .env를 자동으로 읽음

api_key = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI()

# ----------------------------
# Config (env)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


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

# 세션별 업로드 파일 경로들
ATTACHED_PATHS: Dict[str, List[str]] = {}

# ----------------------------
# In-memory storage (demo)
# ----------------------------
SESSIONS = set()
FILES: Dict[str, str] = {}  # file_id -> saved_path
FILE_OWNERS: Dict[str, str] = {}  # file_id -> session_id

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
    execute_tools: bool = False


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
    GPT-4o mini 호출.
    모델 출력은 JSON(assistant_text, action)만.
    action은 서버에서 정책(동의/버튼/데모모드)에 따라 실제 tool_request로 변환.
    """
    developer_instructions = """
        너는 부스용 '문서 요약 비서'다.
        반드시 아래 JSON만 출력해라(추가 텍스트 금지).
        필드:
        - assistant_text: 사용자에게 보여줄 답변(한국어)
        - action: 다음 중 하나
        - "none"
        - "request_camera"   (촬영을 '요청'해야 할 상황)
        - "request_upload"   (전송을 '요청'해야 할 상황)
        - reason: 한 문장 근거(짧게)

        중요:
        - 실제 도구 실행을 지시하지 마라.
        - 개인정보/촬영/전송이 필요하면 반드시 action을 request_* 로만 표시하라.
        """

    # 입력은 message + context_text
    # context_text가 길 수 있으니 너무 길면 앞부분만 보내는 게 좋아요.
    context_trim = context_text[:12000]

    if _openai_client is None:
        return {"assistant_text": "OPENAI_API_KEY가 설정되지 않았어요. .env를 확인해줘요.", "tool_request": []}
    
    resp = _openai_client.responses.create(
        model="gpt-4o-mini",
        # messages 스타일 input (Responses API)
        input=[
            {"role": "developer", "content": developer_instructions},
            {"role": "user", "content": f"사용자 메시지:\n{message}\n\n첨부파일 내용(발췌):\n{context_trim}"},
        ],
        # 모델이 JSON만 내게 강하게 유도
        # (Responses API는 output_text로 텍스트를 받음)
    )

    raw = (resp.output_text or "").strip()

    # 1) JSON 파싱 시도
    try:
        data = json.loads(raw)
    except Exception:
        # 2) 혹시 앞뒤에 잡텍스트가 섞이면 JSON 부분만 대충 추출(방어)
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            # 완전 실패하면 최소 응답으로 폴백
            return {"assistant_text": raw[:500] or "요청을 처리했어요.", "tool_request": []}
        data = json.loads(raw[start : end + 1])

    assistant_text = str(data.get("assistant_text", "")).strip() or "요청을 처리했어요."
    action = str(data.get("action", "none")).strip()
    reason = str(data.get("reason", "")).strip()

    # 여기서는 '실행'이 아니라 '제안'까지만.
    tool_request = []
    if action == "request_camera":
        tool_request.append({"name": "camera_capture", "args": {"reason": reason or "demo"}})
    if action == "request_upload":
        tool_request.append({"name": "send_network_request", "args": {"kind": "image", "reason": reason or "demo"}})

    return {
        "assistant_text": assistant_text,
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
# 서버 살아있나 확인용
@app.get("/health")
def health():
    return {"ok": True, "version": "0.1.0"}

# 세션 발급 + SESSIONS 저장
@app.post("/consent")
def consent():
    session_id = str(uuid.uuid4())
    SESSIONS.add(session_id)

    ATTACHED_PATHS[session_id] = []

    return {"session_id": session_id, "ok": True}

# 세션 유효성 확인
@app.get("/session/{session_id}")
def get_session(session_id: str):
    return {"session_id": session_id, "valid": session_id in SESSIONS}

# 파일 업로드/저장 + 소유자 기록
@app.post("/files")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")

    file_id = uuid.uuid4().hex[:8]
    original_name = file.filename or "upload.bin"
    safe_name = original_name.replace("\\", "_").replace("/", "_")
    save_path = os.path.join(STORAGE_DIR, f"{file_id}_{safe_name}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    FILES[file_id] = save_path
    FILE_OWNERS[file_id] = session_id
    lst = ATTACHED_PATHS.setdefault(session_id, [])
    if save_path not in lst:
        lst.append(save_path)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "mime": file.content_type,
        "size": len(content),
    }

# 세션 소유 파일 목록 조회
@app.get("/files")
def list_files(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")

    items = []
    for fid, path in FILES.items():
        if FILE_OWNERS.get(fid) == session_id:
            items.append({
                "file_id": fid,
                "filename": os.path.basename(path).split("_", 1)[-1],
                "saved_name": os.path.basename(path),
            })
    return {"files": items}

# 사진 찍어서 STORAGE에 저장
@app.post("/tools/camera")
def tool_camera(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")

    try:
        img_path = camera_capture_to_file()
        return {"ok": True, "image": os.path.basename(img_path), "path": img_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 메시지 + 첨부텍스트로 LLM 호출, 응답 돌려줌
@app.post("/chat/send")
def chat_send(data: ChatSendIn):
    # 1) 세션 체크
    if data.session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Unknown session_id. Call /consent first.")
    
    # 2) 파일 텍스트 추출
    attached_paths = ATTACHED_PATHS.get(data.session_id, [])
    context_text = extract_text_from_files(attached_paths)

    # 3) LLM 호출
    llm_out = call_llm(data.message, context_text)

    assistant_text = llm_out.get("assistant_text", "")
    tool_request = llm_out.get("tool_request", [])

    should_execute = DEMO_AUTO_TOOLS or data.execute_tools

    events = []
    events.append({"type": "file_text_extracted", "detail": {"n_files": len(attached_paths)}})
    events.append({"type": "llm_called"})

    if tool_request:
        events.append({"type": "tool_planned", "detail": {"tools": [t.get("name") for t in tool_request]}})

    events.append({"type": "done"})
    return {"assistant": assistant_text, "events": events}


@app.delete("/files/{file_id}")
def delete_file(file_id: str, session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")
    if FILE_OWNERS.get(file_id) != session_id:
        raise HTTPException(status_code=403, detail="forbidden")

    path = FILES.get(file_id)

    # 1) 디스크 파일 삭제
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

    # 2) 메모리 맵 제거
    FILES.pop(file_id, None)
    FILE_OWNERS.pop(file_id, None)

    # 3) 세션의 ATTACHED_PATHS에서도 제거(있으면)
    if path:
        lst = ATTACHED_PATHS.get(session_id, [])
        try:
            lst.remove(path)
        except ValueError:
            pass

    return {"ok": True}


@app.post("/cleanup")
def cleanup_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")

    to_delete = [fid for fid, sid in FILE_OWNERS.items() if sid == session_id]

    deleted = 0
    for fid in to_delete:
        path = FILES.get(fid)

        # 디스크 삭제
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

        FILES.pop(fid, None)
        FILE_OWNERS.pop(fid, None)
        deleted += 1

    ATTACHED_PATHS[session_id] = []

    return {"ok": True, "deleted": deleted}


# - 기존 session_id의 파일/첨부를 정리하고
# - 새로운 session_id를 발급해서 반환
@app.post("/chat/reset")
def chat_reset(data: ResetIn):
    old_sid = data.session_id
    if old_sid not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")

    # 1) 기존 세션이 올린 파일들 삭제(실제 파일 + 메모리)
    to_delete = [fid for fid, sid in FILE_OWNERS.items() if sid == old_sid]
    for fid in to_delete:
        path = FILES.get(fid)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        FILES.pop(fid, None)
        FILE_OWNERS.pop(fid, None)

    # 2) 첨부 경로 목록 제거
    ATTACHED_PATHS.pop(old_sid, None)

    # 3) 기존 세션 제거
    try:
        SESSIONS.remove(old_sid)
    except KeyError:
        pass

    # 4) 새 세션 발급
    new_sid = str(uuid.uuid4())
    SESSIONS.add(new_sid)
    ATTACHED_PATHS[new_sid] = []

    return {"ok": True, "session_id": new_sid, "reset_from": old_sid}


# - storage 내부의 이미지 파일(캡처 등)을 DASHBOARD_UPLOAD_URL로 업로드하는 엔드포인트
# - 존재/확장자도 최소 검증
@app.post("/tools/upload")
def tool_upload(session_id: str, image_path: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")

    abs_storage = os.path.abspath(STORAGE_DIR)
    abs_target = os.path.abspath(image_path)

    # security: storage 밖의 임의 파일 업로드 막기
    if not abs_target.startswith(abs_storage + os.sep) and abs_target != abs_storage:
        raise HTTPException(status_code=400, detail="invalid_image_path")

    if not os.path.exists(abs_target):
        raise HTTPException(status_code=400, detail="image_not_found")

    # 최소한의 파일 타입 체크(원하면 더 엄격히 가능)
    ext = os.path.splitext(abs_target)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="invalid_image_ext")

    try:
        up = upload_image_to_dashboard(abs_target, description="manual run")
        return {"ok": True, "upload": up, "path": abs_target}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
