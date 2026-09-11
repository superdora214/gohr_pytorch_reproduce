@echo off
chcp 65001 >nul
cd /d C:\Code\study\gohr_pytorch_reproduce
set PYTHONIOENCODING=utf-8
echo Generating training curves from the newest log ...
echo.
"C:\Users\lmjkj\anaconda3\envs\pytorch-gpu\python.exe" plot_log.py
if errorlevel 1 (
  echo.
  echo [ERROR] Failed. Read the messages above.
  pause
  exit /b 1
)
echo.
echo Opening training_curves.png ...
start "" training_curves.png
timeout /t 3 >nul
