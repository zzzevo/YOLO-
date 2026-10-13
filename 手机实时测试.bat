@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "LOGFILE=startup_log.txt"

set "CONDA=D:\Anconda\condabin\conda.bat"
set "ENVNAME=yolo26"
set "APPPORT=%~1"
if "%APPPORT%"=="" set "APPPORT=7870"

echo ============================================================
echo   Phone / Camera LIVE TEST  (HTTPS + LAN)
echo.
echo   WHY HTTPS: mobile browsers only allow the camera on a
echo   secure page (https). Plain http:// will show a black box.
echo.
echo   STEPS
echo     1. Phone must be on the SAME Wi-Fi as this PC
echo     2. Open the https address printed below on the phone
echo     3. Certificate warning -^> Advanced -^> Continue
echo     4. Tap the camera box, allow camera, done
echo.
echo   Keep this window OPEN. Closing it stops the service.
echo ============================================================
echo.

>> "%LOGFILE%" echo [%date% %time%] phone live test launcher

if not exist "%CONDA%" (
  echo [ERROR] conda not found: %CONDA%
  echo         Open this file with Notepad and fix CONDA.
  pause
  exit /b 1
)

echo [1/3] Activating conda env %ENVNAME% ...
call "%CONDA%" activate %ENVNAME%
if errorlevel 1 (
  echo [ERROR] conda activate failed. Run: conda env list
  pause
  exit /b 1
)

echo [2/3] Checking the self-signed certificate ...
if exist "certs\cert.pem" (
  echo       certs\cert.pem found - reusing it.
) else (
  echo       not found - will be generated on first start.
)

echo [3/3] Starting workbench with HTTPS (LAN mode) ...
echo.
python -u app.py --lan --browser --port %APPPORT%

echo.
echo Stopped. If something failed, check: %LOGFILE%
>> "%LOGFILE%" echo phone live test exited
pause
