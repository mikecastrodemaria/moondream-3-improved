@echo off
REM Launch the Moondream3 Gradio UI from this app folder.
REM Usage: start.bat
REM Activates the local 'env' venv created by install.bat, then runs app.py.

setlocal
cd /d "%~dp0"

if exist "env\Scripts\activate.bat" (
    call "env\Scripts\activate.bat"
) else (
    echo Virtualenv 'env' not found. Run install.bat first.
    exit /b 1
)

python app.py %*
endlocal
