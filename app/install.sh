#!/usr/bin/env bash
# Install Moondream3 dependencies into a local venv ('env') inside this app folder.
# Cross-platform: macOS (Intel/Apple Silicon), Linux, and Windows Git Bash.
#
# Usage:
#   ./install.sh           # auto-detect platform (CUDA on Linux, MPS/CPU on macOS)
#   ./install.sh --cuda    # force CUDA 12.8 wheel
#   ./install.sh --cpu     # force CPU-only wheel
#
# After install completes, run ./start.sh to launch the Gradio UI.

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Pick a Python binary
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "Error: Python 3.10+ not found on PATH. Install it first."
    exit 1
fi

OS="$(uname -s)"

# Default PyTorch index per platform
case "$OS" in
    Darwin)
        # macOS: official wheel ships MPS support; no CUDA on Mac
        TORCH_INDEX=""
        TORCH_DEFAULT="cpu/mps (macOS)"
        ;;
    Linux)
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        TORCH_DEFAULT="cuda 12.8 (Linux)"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        TORCH_DEFAULT="cuda 12.8 (Windows Git Bash)"
        ;;
    *)
        TORCH_INDEX=""
        TORCH_DEFAULT="cpu (unknown OS: $OS)"
        ;;
esac

# Explicit flag overrides
case "${1:-}" in
    --cpu)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_DEFAULT="cpu (forced)"
        ;;
    --cuda)
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        TORCH_DEFAULT="cuda 12.8 (forced)"
        ;;
esac

echo "Platform: $OS"
echo "PyTorch flavor: $TORCH_DEFAULT"
echo

# Create venv if missing
if [ ! -f "env/bin/activate" ] && [ ! -f "env/Scripts/activate" ]; then
    echo "Creating virtualenv 'env'..."
    "$PY" -m venv env
fi

# Activate (Unix layout or Git-Bash-on-Windows layout)
if [ -f "env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source env/bin/activate
elif [ -f "env/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source env/Scripts/activate
else
    echo "Error: venv was created but no activate script found."
    exit 1
fi

python -m pip install --upgrade pip

if [ -n "$TORCH_INDEX" ]; then
    pip install torch --index-url "$TORCH_INDEX"
else
    # macOS: pull the default PyPI wheel (includes MPS on Apple Silicon)
    pip install torch
fi

pip install -r requirements.txt

echo
echo "Install complete."
echo "Run ./start.sh to launch the Gradio UI."
