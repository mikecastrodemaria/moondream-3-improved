@echo off
REM Launch the Moondream3 batch captioner on a folder of images.
REM Usage: batch_caption.bat [folder] [--overwrite] [--output <captions_folder>] [--online]
REM   --output / -o : Write .txt captions to a different folder (default: same as image folder)
REM   --online      : Allow HuggingFace network access (default: offline, uses local cache only)
REM   Examples:
REM     batch_caption.bat C:\images
REM     batch_caption.bat C:\images --overwrite
REM     batch_caption.bat C:\images --output C:\captions
REM     batch_caption.bat C:\images --online

setlocal
cd /d "%~dp0"

if exist "env\Scripts\activate.bat" (
    call "env\Scripts\activate.bat"
) else (
    echo Virtualenv 'env' not found. Run install first.
    exit /b 1
)

python batch_caption.py %*
endlocal
