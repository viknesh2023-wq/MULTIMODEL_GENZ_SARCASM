@echo off
title Multimodal Sarcasm Intelligence - Frontend UI
echo =======================================================
echo Starting Vite Frontend Dashboard (Port 5173)...
echo URL: http://localhost:5173
echo Frontend Directory: %~dp0frontend
echo =======================================================

cd /d "%~dp0frontend"
call npm run dev
pause
