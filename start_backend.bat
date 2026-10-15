@echo off
title Multimodal Sarcasm Intelligence - Backend API
echo =======================================================
echo Starting FastAPI Backend (Port 8000)...
echo Python Environment: D:\NLP_PROJECT\internvl_env
echo XLM-R Model: D:\NLP_PROJECT\models\xlmr_text_sarcasm_full
echo =======================================================

cd /d D:\NLP_PROJECT\backend
D:\NLP_PROJECT\internvl_env\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
