@echo off
setlocal EnableDelayedExpansion
title Magic Click Launcher

:: ── Check Python availability and version ────────────────────────────────
set "PYTHON_EXE="

:: 1. Check PATH for python or python3
for %%P in (python python3) do (
    where %%P >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON_EXE=%%P"
        goto :check_version
    )
)

:: 2. Check standard Windows installations
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*" "C:\Program Files\Python3*") do (
    if exist "%%D\python.exe" (
        set "PYTHON_EXE=%%D\python.exe"
        goto :check_version
    )
)

if "%PYTHON_EXE%"=="" (
    powershell -command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('Python 3 was not found on this computer.`n`nPlease install Python 3.10+ from python.org (ensure you check `"Add Python to PATH`" during installation), then try again.', 'Magic Click — Python Not Found', 'OK', 'Error')"
    exit /b 1
)

:check_version
for /f "tokens=2" %%V in ('"%PYTHON_EXE%" --version 2^>^&1') do set PY_VER=%%V
for /f "tokens=1,2 delims=." %%A in ("%PY_VER%") do (
    set PY_MAJOR=%%A
    set PY_MINOR=%%B
)
if %PY_MAJOR% LSS 3 (
    powershell -command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('Magic Click requires Python 3.10 or newer.`n`nYou have Python %PY_VER%.`n`nPlease upgrade at python.org.', 'Python Version Error', 'OK', 'Error')"
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    powershell -command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('Magic Click requires Python 3.10 or newer.`n`nYou have Python %PY_VER%.`n`nPlease upgrade at python.org.', 'Python Version Error', 'OK', 'Error')"
    exit /b 1
)

:: ── Change to project root ─────────────────────────────────────────────────
cd /d "%~dp0..\.."

:: ── Show launch confirmation ───────────────────────────────────────────────
set VENV_EXISTS=0
if exist ".venv\Scripts\python.exe" set VENV_EXISTS=1

if "%VENV_EXISTS%"=="1" (
    set MSG=Magic Click is ready to start.^^nYour AI pipeline will open in your browser.
    set BTN_TITLE=Magic Click
) else (
    set MSG=Starting Magic Click for the first time.^^n^^nA one-time setup (~500 MB) is needed. Make sure you are connected to the internet.
    set BTN_TITLE=Magic Click — First Run Setup
)

powershell -command ^
  "Add-Type -AssemblyName PresentationFramework; $r=[System.Windows.MessageBox]::Show('%MSG%', '%BTN_TITLE%', 'OKCancel', 'Information'); if ($r -ne 'OK') { exit 1 }" ^
  || exit /b 0

:: ── Run bootstrap ─────────────────────────────────────────────────────────
"%PYTHON_EXE%" bootstrap.py
if %ERRORLEVEL% NEQ 0 (
    powershell -command "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('Magic Click encountered an error during startup.`n`nPlease check magic_click_setup.log in the Magic Click folder for details.', 'Magic Click — Error', 'OK', 'Error')"
)
endlocal
