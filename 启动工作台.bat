@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "CONDA=D:\Anconda\condabin\conda.bat"
set "ENVNAME=yolo26"
set "APPPORT=%~1"
set "TBPORT=%~2"
if "%APPPORT%"=="" set "APPPORT=7860"
if "%TBPORT%"=="" set "TBPORT=6006"

echo ============================================================
echo   YOLO26 工作台启动器
echo   工作台    : http://127.0.0.1:%APPPORT%/
echo   训练曲线  : http://127.0.0.1:%TBPORT%/
echo ============================================================
echo.

if not exist "%CONDA%" (
  echo [错误] 找不到 conda: %CONDA%
  echo         请修改本文件里的 CONDA 变量。
  pause
  exit /b 1
)

echo [1/2] 打开 TensorBoard 窗口（可最小化，不影响使用）...
start "TensorBoard (%TBPORT%)" cmd /k "chcp 65001 >nul && call "%CONDA%" activate %ENVNAME% && cd /d "%~dp0" && python -m tensorboard.main --logdir runs --port %TBPORT% --host 127.0.0.1"

echo [2/2] 启动工作台（本窗口保持开启，关掉即停止服务）...
call "%CONDA%" activate %ENVNAME%
python app.py --port %APPPORT%

echo.
echo 服务已退出。
pause
