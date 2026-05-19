#!/usr/bin/env bash
# Launch the Moondream3 batch captioner on a folder of images.
# Usage:
#   ./batch_caption.sh [folder] [--overwrite] [--output <captions_folder>] [--online]
#   --output / -o : Write .txt captions to a different folder (default: same as image folder)
#   --online      : Allow HuggingFace network access (default: offline, uses local cache only)
# Examples:
#   ./batch_caption.sh /path/to/images
#   ./batch_caption.sh /path/to/images --overwrite
#   ./batch_caption.sh /path/to/images --output /path/to/captions
#   ./batch_caption.sh /path/to/images --online
# If no folder is given, the script will prompt for one.

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Activate venv (cross-platform: Linux/macOS or Windows Git Bash)
if [ -f "env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source env/bin/activate
elif [ -f "env/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source env/Scripts/activate
else
    echo "Virtualenv 'env' not found. Run install first."
    exit 1
fi

python batch_caption.py "$@"
