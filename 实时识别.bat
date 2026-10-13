@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "LOGFILE=startup_log.txt"

set "CONDA=D:\Anconda\condabin\conda.bat"
set "ENVNAME=yolo26"
set "APPPORT=%~1"
set "MODEL=%~2"
if "%APPPORT%"=="" set "APPPORT=7871"
if "%MODEL%"=="" set "MODEL=runs\ball_v11\weights\best.onnx"

echo ============================================================
echo   REALTIME DETECTION  (boxes drawn ON the live camera view)
echo.
echo   - Boxes are drawn directly on the phone live picture and
echo     keep following the ball (smoothed tracking + short coast)
echo   - Inference runs on THIS PC. Phone only shows the picture.
echo.
echo   STEPS
echo     1. Phone on the SAME Wi-Fi as this PC
echo     2. Open the https address printed below (or scan the QR
echo        shown on the PC browser), accept the certificate
echo     3. Tap the blue "start camera" button and allow camera
echo.
echo   Tips
echo     - Faster: pass a .pt model as 2nd arg, e.g.
echo         %~nx0 7871 runs\ball_v11\weights\best.pt
echo     - Keep this window OPEN. Closing it stops the service.
echo ============================================================
echo.

>> "%LOGFILE%" echo [%date% %time%] live overlay launcher model=%MODEL%

if not exist "%CONDA%" (
  echo [ERROR] conda not found: %CONDA%
  pause
  exit /b 1
)

call "%CONDA%" activate %ENVNAME%
if errorlevel 1 (
  echo [ERROR] conda activate failed. Run: conda env list
  pause
  exit /b 1
)

echo Starting realtime server on port %APPPORT% ...
echo.
python -u live_overlay.py --port %APPPORT% --model "%MODEL%" --open

echo.
echo Stopped. Check %LOGFILE% if something failed.
>> "%LOGFILE%" echo live overlay exited
pause
