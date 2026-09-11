@echo off
chcp 65001 >nul
cd /d C:\Code\study\gohr_pytorch_reproduce
echo Starting TensorBoard ...
echo Open in browser: http://localhost:6006
echo Close this window to stop.
echo.
"C:\Users\lmjkj\anaconda3\envs\pytorch-gpu\python.exe" -m tensorboard.main --logdir checkpoints --port 6006
pause
