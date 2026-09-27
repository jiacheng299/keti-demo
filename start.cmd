@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\download_models.py --check
if errorlevel 1 (
    echo Please run setup.cmd to prepare the model files.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
if errorlevel 1 (
    pause
    exit /b 1
)
