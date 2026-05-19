# 🌙 Moondream3 — Improved Gradio UI

A polished, batch-ready web interface for the [Moondream3](https://huggingface.co/moondream/moondream3-preview) vision-language model (9B total params, 2B active MoE), with image captioning, visual Q&A, object detection, pointing, and a brand-new **folder-level batch captioning** workflow.

> This fork extends the original [MoonDream-3-Pinokio](https://github.com/PierrunoYT/MoonDream-3-Pinokio) launcher with a richer UI, a batch dataset-labeling pipeline, a HuggingFace cache panel, and cross-platform standalone scripts.

---

## ✨ What's new in this version

| | Feature | Notes |
|-|---------|-------|
| 📁 | **Batch Caption tab** | Process a whole folder of images from inside the UI — folder path **+** drag-and-drop hybrid, optional output folder, choice of caption length, overwrite toggle, live progress log. |
| 🛠️ | **CLI batch captioner** | `batch_caption.py` + Windows `.bat` / Unix `.sh` wrappers for headless / scripted use. Sidecar `.txt` next to each image, skip-if-exists, offline by default. |
| 💾 | **HuggingFace Cache panel** | New "💾 HF Cache" tab: shows active cache path, whether weights are cached, total cache size, offline status. Toggle `HF_HUB_OFFLINE`, override `HF_HOME`, open the cache folder, refresh — all from the browser. |
| 🚀 | **Standalone launchers** | Cross-platform `install.sh` / `start.sh` / `install.bat` / `start.bat` in `app/` — auto-detect platform (CUDA on Linux, MPS/CPU on macOS) and manage a local venv. No Pinokio required. |
| 🩹 | **Pinokio reliability fixes** | `start.js` now uses unbuffered Python output, a stricter URL-capture regex, and `GRADIO_SERVER_NAME=127.0.0.1` so the Pinokio "Open Web UI" button reliably picks up the running server. |
| 🧹 | **Cleaner repo** | `.gitattributes` enforces LF endings on shell scripts, `.gitignore` excludes AI-assistant artifacts and internal notes. |

---

## Features

- **📝 Image Captioning** — short, normal, or long captions with optional token-by-token streaming
- **❓ Visual Q&A** — ask questions about an image with reasoning mode (also supports text-only queries)
- **🔍 Object Detection** — find and box objects by name with adjustable max-objects
- **👆 Object Pointing** — pinpoint objects with circular markers
- **📁 Batch Caption** *(new)* — hybrid folder-path + drag-drop, optional output dir, length selector, live progress
- **💾 HF Cache panel** *(new)* — inspect / override / open / toggle offline mode for the HuggingFace cache
- **⚙️ Advanced settings** — temperature, max tokens, streaming, max objects

---

## Requirements

- Python **3.10+**
- NVIDIA GPU with CUDA recommended (~19 GB VRAM for full performance)
- Also works on **Apple Silicon (MPS)** and **CPU** — slower but functional
- ~18 GB free disk space for the model weights (first run only)

---

## Installation

### Option A — One-click via Pinokio *(recommended)*

1. Install [Pinokio](https://pinokio.computer/).
2. Open Pinokio → **Discover** → paste this repo URL → **Download / Install**.
3. Click **Install** in the launcher UI, wait for dependencies to finish, then click **Start** → **Open Web UI**.

### Option B — Standalone (no Pinokio)

#### macOS / Linux / Git Bash

```bash
git clone <your-repo-url>
cd moondream-3-improved/app
chmod +x install.sh start.sh batch_caption.sh
./install.sh            # auto-detects CUDA (Linux) or MPS/CPU (macOS)
# ./install.sh --cpu    # force CPU-only
# ./install.sh --cuda   # force CUDA 12.8
./start.sh
```

#### Windows (CMD / PowerShell)

```bat
git clone <your-repo-url>
cd moondream-3-improved\app
install.bat             REM defaults to CUDA 12.8; use `install.bat --cpu` for CPU-only
start.bat
```

Then open the URL printed in the terminal (typically `http://127.0.0.1:7860`).

### Option C — Manual

```bash
python -m venv .venv
source .venv/bin/activate           # or .venv\Scripts\activate on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or .../whl/cpu
pip install -r app/requirements.txt
python app/app.py
```

---

## Usage

1. Open the URL in your browser.
2. Click **🚀 Load Model** — first run downloads ~18 GB; subsequent runs read from disk.
3. Pick a tab:
   - **📝 Image Captioning** — upload, choose length, generate.
   - **❓ Visual Q&A** — upload (optional), ask a question, toggle reasoning.
   - **🔍 Object Detection** — upload, name the object, set max count.
   - **👆 Object Pointing** — upload, name the object.
   - **📁 Batch Caption** — paste a folder path **and/or** drag-drop images, optionally set an output folder and length, click "Caption All Images".
   - **💾 HF Cache** — inspect cache, toggle offline mode, override cache path, open in file explorer.

### Batch Caption — CLI

For headless / scripted dataset labeling:

```bash
# Direct
python app/batch_caption.py <folder> [--overwrite] [--output <captions_folder>] [--online]

# Windows wrapper (activates the local venv automatically)
app\batch_caption.bat C:\path\to\images
app\batch_caption.bat C:\path\to\images --overwrite
app\batch_caption.bat C:\path\to\images --output C:\path\to\captions

# macOS / Linux / Git Bash
./app/batch_caption.sh /path/to/images --output /path/to/captions
```

Supported extensions: `jpg, jpeg, png, bmp, gif, webp, tiff, tif`. Re-runs skip images that already have a matching `.txt` unless `--overwrite` is passed. The CLI runs **offline by default** once the model is cached — add `--online` to allow HuggingFace network checks for a new revision.

### Advanced settings

- **Temperature** — 0 = model default, higher = more random
- **Max Tokens** — 0 = model default
- **Stream Output** — token-by-token streaming for captioning / Q&A
- **Reasoning** — chain-of-thought for Q&A
- **Max Objects** — cap detection count

---

## 🔭 Coming next

A short teaser of what's on the roadmap (full backlog tracked privately):

- **Resume-safe batch jobs** — JSONL manifest so a crashed batch can pick up where it left off
- **Recursive folder scan + glob filters** (`--recursive`, `--include "*.jpg"`)
- **Custom prompt + temperature in batch mode** — dramatically better dataset captions
- **Parallel image decoding** — overlap CPU image-loading with GPU inference for ~10–20% throughput on large folders
- **Batch Q&A and Batch Detection** — not just captions: dataset-wide classification / labeling
- **Dataset-export presets** — one-click sidecar formats for Kohya / LoRA / SD training pipelines
- **Auto-offline detection** — startup green badge when weights are already fully cached
- **OOM-safe retries** — catch CUDA OOM, free memory, retry at lower resolution

Have a feature in mind? Open an issue or PR.

---

## License

See the [Moondream3 Model Card](https://huggingface.co/moondream/moondream3-preview) for model license information. Launcher / UI code in this repo is released under the same terms as the upstream [PierrunoYT/MoonDream-3-Pinokio](https://github.com/PierrunoYT/MoonDream-3-Pinokio) project.

---

*Powered by [Moondream3](https://huggingface.co/moondream/moondream3-preview) & [Gradio](https://gradio.app) — Pinokio launcher conventions courtesy of [Pinokio](https://pinokio.computer/).*
