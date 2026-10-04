@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "CONDA=D:\Anconda\condabin\conda.bat"
set "ENVNAME=yolo26"
set "TBPORT=%~1"
if "%TBPORT%"=="" set "TBPORT=6006"

echo TensorBoard URL: http://127.0.0.1:%TBPORT%/
call "%CONDA%" activate %ENVNAME%
python -m tensorboard.main --logdir runs --port %TBPORT% --host 127.0.0.1

echo.
echo TensorBoard stopped.
pause
