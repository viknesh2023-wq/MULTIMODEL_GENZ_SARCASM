@echo off
title Multimodal Sarcasm Intelligence - Backend API
echo =======================================================
echo Starting FastAPI Backend (Port 8000)...
echo =======================================================

cd /d "%~dp0backend"

if exist "%~dp0internvl_env\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0internvl_env\Scripts\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo Python Runtime: %PYTHON_EXE%
echo Backend Directory: %~dp0backend
echo =======================================================
"%PYTHON_EXE%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
