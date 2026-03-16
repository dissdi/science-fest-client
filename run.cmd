@echo off
setlocal

set BASE=%~dp0
set SERVER_DIR=%BASE%server
set CLIENT_DIR=%BASE%client
set GUI_DIR=%BASE%gui

set SERVER_PORT=8000
set CLIENT_PORT=5177

start "SERVER" cmd /k "cd /d "%SERVER_DIR%" && python -m uvicorn main:app --reload --port %SERVER_PORT%"
start "CLIENT" cmd /k "cd /d "%CLIENT_DIR%" && python -m uvicorn agent_server:app --reload --port %CLIENT_PORT%"
start "GUI"    cmd /k "cd /d "%GUI_DIR%" && python -m streamlit run main/app.py"

endlocal