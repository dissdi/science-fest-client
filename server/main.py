from fastapi import Form
from fastapi.responses import RedirectResponse

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import aiofiles
import cv2
import numpy as np
import os
import json
from datetime import datetime

app = FastAPI()

UPLOAD_DIR = "uploads"
LOG_FILE = "hack_log.json"
COUNT_FILE = "hack_count.txt"

templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Initialize log and count files if not exist
def init_files():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)
    if not os.path.exists(COUNT_FILE):
        with open(COUNT_FILE, "w") as f:
            f.write("0")

init_files()

# Helper to update hack count
def increment_hack_count():
    count = 0
    if os.path.exists(COUNT_FILE):
        with open(COUNT_FILE, "r") as f:
            try:
                count = int(f.read())
            except:
                count = 0
    count += 1
    with open(COUNT_FILE, "w") as f:
        f.write(str(count))
    return count

def get_hack_count():
    if os.path.exists(COUNT_FILE):
        with open(COUNT_FILE, "r") as f:
            try:
                return int(f.read())
            except:
                return 0
    return 0

# Helper to log events
def log_event(event):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    logs.append(event)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
@app.post("/delete_image")
async def delete_image(filename: str = Form(...)):
    # Mark image as deleted in log
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    updated = False
    for log in logs:
        if log.get("type") == "image" and log.get("filename") == filename and not log.get("deleted"):
            log["deleted"] = True
            updated = True
    if updated:
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        # Optionally, delete the actual file
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
    return RedirectResponse(url="/dashboard", status_code=303)
@app.post(
    "/api/upload/image",
    summary="이미지(사진) 업로드",
    description="Agent가 해킹 사진을 업로드할 때 사용합니다. multipart/form-data로 이미지 파일(file)과 설명(description)을 전송하세요."
)
async def upload_image(
    file: UploadFile = File(..., description="업로드할 이미지 파일 (예: victim_face.jpg)"),
    description: str = Form(None, description="상황 설명 (선택)")
):
    timestamp = datetime.now().isoformat()
    filename = f"{timestamp.replace(':', '-')}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    content = await file.read()
    # Mosaic image before saving
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        # Mosaic: resize down then up
        h, w = img.shape[:2]
        mosaic_scale = 0.05  # 5% size
        small = cv2.resize(img, (max(1,int(w*mosaic_scale)), max(1,int(h*mosaic_scale))), interpolation=cv2.INTER_LINEAR)
        mosaic_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode('.jpg', mosaic_img)
        mosaic_bytes = buf.tobytes()
        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(mosaic_bytes)
    else:
        # If not an image, just save as is
        async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(content)
    event = {
        "type": "image",
        "filename": filename,
        "description": description,
        "timestamp": timestamp
    }
    log_event(event)
    count = increment_hack_count()
    return {"status": "success", "count": count}

@app.post(
    "/api/upload/text",
    summary="텍스트 정보 업로드",
    description="Agent가 해킹 관련 텍스트/명령을 업로드할 때 사용합니다. application/json 형식의 자유로운 데이터를 전송하세요."
)
async def upload_text(request: Request):
    data = await request.json()
    timestamp = datetime.now().isoformat()
    event = {
        "type": "text",
        "data": data,
        "timestamp": timestamp
    }
    log_event(event)
    count = increment_hack_count()
    return {"status": "success", "count": count}

@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="해킹 대시보드",
    description="해킹된 이미지, 텍스트, 해킹 횟수를 웹페이지로 시각화합니다. 브라우저에서 직접 접속하세요."
)
async def dashboard(request: Request):
    # Load logs and count
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    count = get_hack_count()
    return templates.TemplateResponse("dashboard.html", {"request": request, "logs": logs, "count": count})

def main():
    print("Hello from science-fest-server!")


if __name__ == "__main__":
    main()
