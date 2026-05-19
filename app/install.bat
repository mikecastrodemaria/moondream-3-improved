@echo off
REM Install Moondream3 dependencies into a local venv ('env') inside this app folder.
REM Usage: install.bat [--cuda | --cpu]
REM   --cuda  : Install PyTorch with CUDA 12.8 (default if no flag)
REM   --cpu   : Install CPU-only PyTorch
REM
REM After install completes, run start.bat to launch the Gradio UI.

setlocal
cd /d "%~dp0"

set TORCH_INDEX=https://download.pytorch.org/whl/cu128
if /I "%~1"=="--cpu" set TORCH_INDEX=https://download.pytorch.org/whl/cpu
if /I "%~1"=="--cuda" set TORCH_INDEX=https://download.pytorch.org/whl/cu128

if not exist "env\Scripts\activate.bat" (
    echo Creating virtualenv 'env'...
    python -m venv env
    if errorlevel 1 (
        echo Failed to create venv. Make sure Python 3.10+ is on PATH.
        exit /b 1
    )
)

call "env\Scripts\activate.bat"

python -m pip install --upgrade pip
pip install torch --index-url %TORCH_INDEX%
pip install -r requirements.txt

echo.
echo Install complete. Run start.bat to launch the UI.
endlocal
