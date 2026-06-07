@echo off
rem ===========================================================================
rem  tracevault - one-time installer: creates Desktop + Start-menu shortcuts
rem  that launch the native desktop app. No admin required (per-user).
rem  To also open tracevault at login:  uv run tracevault autostart enable
rem ===========================================================================
setlocal
pushd "%~dp0"
set "REPO=%CD%"

powershell -NoProfile -NonInteractive -Command "$ws=New-Object -ComObject WScript.Shell; $t=Join-Path '%REPO%' 'tracevault.cmd'; foreach($d in @($ws.SpecialFolders('Desktop'),$ws.SpecialFolders('Programs'))){ $lnk=Join-Path $d 'tracevault.lnk'; $s=$ws.CreateShortcut($lnk); $s.TargetPath=$t; $s.WorkingDirectory='%REPO%'; $s.WindowStyle=7; $s.Description='tracevault - local knowledge base'; $s.Save() }"

if errorlevel 1 (
  echo Failed to create shortcuts.
  popd & endlocal & exit /b 1
)
echo Done. tracevault shortcuts are on your Desktop and in the Start menu.
echo Double-click "tracevault" to launch it.
popd
endlocal & exit /b 0
