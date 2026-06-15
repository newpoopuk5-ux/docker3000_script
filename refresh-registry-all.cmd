@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python not found on PATH. Install Python 3 or use Comfy portable python.
  exit /b 1
)

echo.
echo === Muse model registry: Civitai refresh all ===
echo Folder: %CD%
echo.

python refresh_registry.py --all
set EXIT=%ERRORLEVEL%

if %EXIT% neq 0 (
  echo.
  echo Refresh failed with exit code %EXIT%.
  exit /b %EXIT%
)

echo.
echo === Re-sync index from roots (keeps all refreshed versions) ===
python build_registry.py
set EXIT=%ERRORLEVEL%

if %EXIT% neq 0 (
  echo.
  echo build_registry.py failed with exit code %EXIT%.
  exit /b %EXIT%
)

echo.
echo All done. Commit muse_worker\catalog\registry\*.json when ready.
echo.
pause
