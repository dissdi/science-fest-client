@echo off
setlocal

set BASE=%~dp0
set SERVER_DIR=%BASE%server
set SERVER_PORT=8000

start "SERVER" cmd /k "cd /d "%SERVER_DIR%" && python -m uvicorn main:app --reload --host 0.0.0.0 --port %SERVER_PORT%"

endlocal