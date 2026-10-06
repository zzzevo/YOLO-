@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "CONDA=D:\Anconda\condabin\conda.bat"
set "ENVNAME=labeling"
set "REVIEW=%~1"
if "%REVIEW%"=="" set "REVIEW=%~dp0review"

echo ============================================================
echo   X-AnyLabeling  (YOLO annotation tool)
echo.
echo   images : %REVIEW%\images
echo   labels : %REVIEW%\labels
echo.
echo   First launch takes 10-30 seconds. Keep this window open.
echo   Closing this window closes the tool.
echo ============================================================
echo.

if not exist "%CONDA%" (
  echo [ERROR] conda not found: %CONDA%
  echo         Edit this file and fix the CONDA variable.
  pause
  exit /b 1
)
if not exist "%REVIEW%\images" (
  echo [ERROR] image folder not found: %REVIEW%\images
  echo         Run first:  python prepare_review.py --src raw_ball --dst dataset_ball --review review
  pause
  exit /b 1
)

call "%CONDA%" activate %ENVNAME%
xanylabeling --filename "%REVIEW%\images" --output "%REVIEW%\labels" --no-auto-update-check

echo.
echo X-AnyLabeling closed.
pause
