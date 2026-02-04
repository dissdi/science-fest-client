from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os, uuid

app = FastAPI()

# 아주 단순한 메모리 저장(부스 데모용)
SESSIONS = set()
FILES = {}  # file_id -> saved_path

STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

class ConsentIn(BaseModel):
    consent: bool

class ChatSendIn(BaseModel):
    session_id: str
    message: str
    file_ids: list[str] = []

class ResetIn(BaseModel):
    session_id: str

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

    # 첨부 파일 경로 확인
    attached_paths = []
    for fid in data.file_ids:
        if fid not in FILES:
            raise HTTPException(status_code=400, detail=f"unknown_file_id: {fid}")
        attached_paths.append(FILES[fid])

    # TODO: 여기서 message + attached_paths를 가지고
    # - 파일 텍스트 추출
    # - LLM 호출
    # - (동의 범위 내) 카메라 캡처
    # - 대시보드 서버로 업로드
    # 를 수행하고 결과를 반환

    return {
        "assistant": f"(데모 응답) 메시지 받음: {data.message} / 첨부 {len(attached_paths)}개",
        "events": [{"type": "analysis_started"}],
    }

@app.post("/chat/reset")
def chat_reset(data: ResetIn):
    if data.session_id not in SESSIONS:
        raise HTTPException(status_code=403, detail="consent_required")
    # 데모용: 세션 유지하면서 첨부만 초기화하고 싶으면 여기를 바꿔도 됨
    return {"ok": True}
