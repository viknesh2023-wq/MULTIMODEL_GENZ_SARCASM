@echo off
title Multimodal Sarcasm Intelligence - Frontend UI
echo =======================================================
echo Starting Vite Frontend Dashboard (Port 5173)...
echo URL: http://localhost:5173
echo =======================================================

set PATH=C:\Program Files\nodejs;%PATH%
cd /d D:\NLP_PROJECT\frontend
call npm run dev
pause
