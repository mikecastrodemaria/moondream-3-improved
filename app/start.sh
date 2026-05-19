#!/usr/bin/env bash
# Launch the Moondream3 Gradio UI from this app folder.
# Usage: ./start.sh
# Activates the local 'env' venv created by install.sh, then runs app.py.

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ -f "env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source env/bin/activate
elif [ -f "env/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source env/Scripts/activate
else
    echo "Virtualenv 'env' not found. Run ./install.sh first."
    exit 1
fi

python app.py "$@"
