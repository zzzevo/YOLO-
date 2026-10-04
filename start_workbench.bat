@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "LOGFILE=startup_log.txt"

set "CONDA=D:\Anconda\condabin\conda.bat"
set "ENVNAME=yolo26"
set "APPPORT=%~1"
set "TBPORT=%~2"
if "%APPPORT%"=="" set "APPPORT=7860"
if "%TBPORT%"=="" set "TBPORT=6006"

echo ============================================================
echo   YOLO26 Workbench Launcher
echo.
echo   Workbench : http://127.0.0.1:%APPPORT%/
echo   Curves    : http://127.0.0.1:%TBPORT%/
echo.
echo   It is a WEB UI - the browser opens automatically.
echo   Keep this window OPEN. Closing it stops the service.
echo ============================================================
echo.

> "%LOGFILE%" echo [%date% %time%] launcher start app=%APPPORT% tb=%TBPORT%

if not exist "%CONDA%" (
  echo [ERROR] conda not found: %CONDA%
  echo         Open this file with Notepad and fix the CONDA variable.
  >> "%LOGFILE%" echo [ERROR] conda not found
  echo.
  pause
  exit /b 1
)

rem ---------- case 1: already running -> only open the browser ----------
curl -s -o NUL --max-time 3 http://127.0.0.1:%APPPORT%/ >nul 2>&1
if not errorlevel 1 (
  echo [i] Workbench is already running on port %APPPORT%.
  echo     Opening browser only. This window closes in a few seconds.
  >> "%LOGFILE%" echo [i] reuse running instance
  start "" http://127.0.0.1:%APPPORT%/
  start "" http://127.0.0.1:%TBPORT%/
  ping -n 6 127.0.0.1 >nul
  exit /b 0
)

echo [1/3] Starting TensorBoard window ...
start "TensorBoard %TBPORT%" "%~dp0start_tensorboard.bat" %TBPORT%
>> "%LOGFILE%" echo [1/3] tensorboard window started

ping -n 4 127.0.0.1 >nul

echo [2/3] Activating conda env %ENVNAME% ...
call "%CONDA%" activate %ENVNAME%
if errorlevel 1 (
  echo [ERROR] conda activate failed. Check with: conda env list
  >> "%LOGFILE%" echo [ERROR] conda activate failed
  echo.
  pause
  exit /b 1
)
>> "%LOGFILE%" echo [2/3] env activated

echo [3/3] Starting workbench - the browser opens when it is ready.
>> "%LOGFILE%" echo [3/3] start app.py
python app.py --port %APPPORT% --browser

echo.
echo Workbench stopped. If there were errors above, send me a screenshot
echo or check the file: %LOGFILE%
>> "%LOGFILE%" echo workbench exited
pause
