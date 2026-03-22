@echo off
setlocal

set BASE=%~dp0
set CLIENT_DIR=%BASE%client
set GUI_DIR=%BASE%gui

set CLIENT_PORT=5177

start "CLIENT" cmd /k "cd /d "%CLIENT_DIR%" && python -m uvicorn agent_server:app --reload --host 0.0.0.0 --port %CLIENT_PORT%"
start "GUI"    cmd /k "cd /d "%GUI_DIR%" && python -m streamlit run main/app.py"

endlocal