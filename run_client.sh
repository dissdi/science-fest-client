#!/bin/bash

BASE="$(cd "$(dirname "$0")" && pwd)"
CLIENT_DIR="$BASE/client"
GUI_DIR="$BASE/gui"

CLIENT_PORT=5177

osascript -e "tell app \"Terminal\" to do script \"cd '$CLIENT_DIR' && python -m uvicorn agent_server:app --reload --host 0.0.0.0 --port $CLIENT_PORT\""
osascript -e "tell app \"Terminal\" to do script \"cd '$GUI_DIR' && python -m streamlit run main/app.py\""