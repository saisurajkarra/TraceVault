@echo off
rem ===========================================================================
rem  tracevault - double-click launcher (zero-Docker native desktop app).
rem  Brings up the app in its own window. `tracevault desktop` starts a real
rem  local MinIO server itself, so Docker is NOT required. First run downloads
rem  the local models + the MinIO binary (one time); later runs are instant.
rem ===========================================================================
setlocal
pushd "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo.
  echo  uv was not found on your PATH. Install it from https://docs.astral.sh/uv/
  echo  then double-click this file again.
  echo.
  pause
  popd & endlocal & exit /b 1
)

echo Starting tracevault... (first run downloads local models + MinIO, one time)
uv run --extra desktop tracevault desktop
set "ERR=%ERRORLEVEL%"

popd
if not "%ERR%"=="0" (
  echo.
  echo tracevault exited with an error ^(code %ERR%^). See the messages above.
  pause
)
endlocal & exit /b %ERR%
