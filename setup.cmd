@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
py -3.13 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    py -3.13 scripts\setup_demo.py
) else (
    python scripts\setup_demo.py
)
if errorlevel 1 (
    echo Setup failed. Install Python 3.13 64-bit and check the error above.
    pause
    exit /b 1
)
pause
