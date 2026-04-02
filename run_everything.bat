@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo ==========================================
echo          Project Alpha Launcher
echo ==========================================

goto :menu

:menu
echo.
echo [1] Setup venv + install dependencies
echo [2] Run Voice Assistant (main_chat.py)
echo [3] Setup + Run Voice Assistant
echo [4] Exit
choice /c 1234 /n /m "Choose an option: "
if errorlevel 4 goto :end
if errorlevel 3 goto :setup_and_run
if errorlevel 2 goto :run_assistant
if errorlevel 1 goto :setup_only

goto :menu

:resolve_python
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.10 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.10"
    if not defined PY_CMD py -3.11 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.11"
    if not defined PY_CMD py -3.12 -c "import sys" >nul 2>nul && set "PY_CMD=py -3.12"
)
if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Compatible Python was not found in PATH.
    echo Install Python 3.10, 3.11, or 3.12 and reopen terminal.
    exit /b 1
)
exit /b 0

:ensure_venv
if exist ".venv\Scripts\python.exe" (
    set "VENV_PY=.venv\Scripts\python.exe"
    for /f "tokens=2" %%v in ('"%VENV_PY%" --version') do set "VENV_VER=%%v"
    for /f "tokens=1,2 delims=." %%a in ("!VENV_VER!") do (
        set "VENV_MAJOR=%%a"
        set "VENV_MINOR=%%b"
    )

    rem numpy<2.0 has no wheel for Python >=3.13 in this setup, so recreate with compatible Python.
    if !VENV_MAJOR! GEQ 3 if !VENV_MINOR! GEQ 13 (
        echo [SETUP] Existing .venv uses Python !VENV_VER! which is incompatible with requirements.
        echo [SETUP] Recreating .venv with Python 3.10/3.11/3.12...
        rmdir /s /q .venv
    ) else (
        exit /b 0
    )
)

call :resolve_python
if errorlevel 1 exit /b 1

echo [SETUP] Creating virtual environment using %PY_CMD%...
%PY_CMD% -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create .venv
    exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"
for /f "tokens=2" %%v in ('"%VENV_PY%" --version') do set "NEW_VENV_VER=%%v"
for /f "tokens=1,2 delims=." %%a in ("!NEW_VENV_VER!") do (
    set "NEW_VENV_MAJOR=%%a"
    set "NEW_VENV_MINOR=%%b"
)
if !NEW_VENV_MAJOR! GEQ 3 if !NEW_VENV_MINOR! GEQ 13 (
    echo [ERROR] Created .venv with Python !NEW_VENV_VER!, but requirements need Python 3.10-3.12.
    echo [HINT] Install Python 3.10/3.11 and rerun this script.
    exit /b 1
)
exit /b 0

:install_deps
call :ensure_venv
if errorlevel 1 exit /b 1

echo [SETUP] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

echo [SETUP] Installing requirements.txt...
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.txt
    exit /b 1
)

if exist "extra-req.txt" (
    echo [SETUP] Installing extra-req.txt...
    "%VENV_PY%" -m pip install -r extra-req.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install extra-req.txt
        exit /b 1
    )
)

echo [OK] Setup complete.
exit /b 0

:ensure_openrouter
echo.
if defined OPENROUTER_API_KEY exit /b 0
echo [INFO] OPENROUTER_API_KEY is not set for this terminal session.
set /p OPENROUTER_API_KEY=Paste OPENROUTER_API_KEY (or press Enter to skip): 
if "%OPENROUTER_API_KEY%"=="" (
    echo [WARN] No key entered. LLM may be unavailable.
) else (
    echo [OK] Key set for this session.
)
exit /b 0

:setup_only
echo.
call :install_deps
if errorlevel 1 goto :pause_menu

goto :pause_menu

:run_assistant
echo.
call :ensure_venv
if errorlevel 1 goto :pause_menu
call :ensure_openrouter

echo [RUN] Starting main_chat.py...
"%VENV_PY%" main_chat.py
goto :pause_menu

:setup_and_run
echo.
call :install_deps
if errorlevel 1 goto :pause_menu
call :ensure_openrouter

echo [RUN] Starting main_chat.py...
"%VENV_PY%" main_chat.py
goto :pause_menu

:pause_menu
echo.
pause
goto :menu

:end
echo.
echo Bye.
endlocal
exit /b 0
