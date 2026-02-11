1. 동의 /consent
2. 파일 업로드 /files (form에 session_id 포함)
3. 요약 요청 /chat/send → tool_planned 나오면
4. 진행자가 버튼으로:
    “촬영” → /tools/camera
    “전송” → /tools/upload

실행
server 1. python -m uvicorn main:app --reload --port 8000
client 2. python -m uvicorn agent_server:app --reload --port 5177
gui    3. python -m streamlit run main/app.py

.env
OPENAI_API_KEY=
DASHBOARD_UPLOAD_URL=http://127.0.0.1:8000/api/upload/image
ALLOWED_UPLOAD_HOSTS=127.0.0.1,localhost
DUMMY_CAMERA=false or true (practice / test)
CAM_INDEX=1 (0, 1, 2 check which one works)