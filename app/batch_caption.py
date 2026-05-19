"""
Batch caption every image in a folder using Moondream3.
Writes <image_basename>.txt next to each image with the "long" caption.

Usage:
    python batch_caption.py <folder> [--overwrite] [--output <captions_folder>] [--online]

By default runs in offline mode (HF_HUB_OFFLINE=1) so no network calls are made
once the model is cached. Pass --online to allow HuggingFace network access
(e.g. to pull a new model revision).
"""

import argparse
import os
import sys
import warnings

os.environ["TORCH_COMPILE_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Default to offline mode — the model is cached after the first run.
# Can be overridden via the --online flag (handled in main()) or by
# pre-setting HF_HUB_OFFLINE in the environment.
if "--online" not in sys.argv and os.environ.get("HF_HUB_OFFLINE") is None:
    os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM
from PIL import Image

try:
    from torch.nn.attention.flex_attention import BlockMask
    _orig_init = BlockMask.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if not hasattr(self, "seq_lengths"):
            if hasattr(self, "shape"):
                self.seq_lengths = self.shape
            elif hasattr(self, "kv_num_blocks"):
                self.seq_lengths = (self.kv_num_blocks[-1] * 128, self.kv_num_blocks[-1] * 128)
            else:
                self.seq_lengths = None

    BlockMask.__init__ = _patched_init
except Exception:
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def main():
    parser = argparse.ArgumentParser(description="Batch-caption a folder with Moondream3.")
    parser.add_argument("folder", nargs="?", help="Folder containing images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    parser.add_argument("-o", "--output", default=None,
                        help="Folder to write .txt captions into (default: same as image folder)")
    parser.add_argument("--online", action="store_true",
                        help="Allow HuggingFace network access (default: offline, uses local cache only)")
    args = parser.parse_args()

    if args.online:
        os.environ["HF_HUB_OFFLINE"] = "0"
        print("Online mode: HuggingFace network access enabled.")
    else:
        print("Offline mode: using local HuggingFace cache only (pass --online to allow downloads).")

    folder = args.folder
    if not folder:
        folder = input("Folder path: ").strip().strip('"').strip("'")

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output) if args.output else folder
    os.makedirs(output_dir, exist_ok=True)

    images = [
        f for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        and os.path.isfile(os.path.join(folder, f))
    ]
    if not images:
        print(f"No images found in: {folder}")
        sys.exit(0)

    device, dtype = pick_device()
    print(f"Loading Moondream3 on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        dtype=dtype,
        device_map={"": device},
    )
    try:
        model.compile()
    except Exception as e:
        print(f"Compile skipped: {e}")

    total = len(images)
    done = skipped = failed = 0
    print(f"Found {total} image(s) in {folder}")
    if output_dir != folder:
        print(f"Writing captions to {output_dir}")

    for idx, name in enumerate(images, start=1):
        img_path = os.path.join(folder, name)
        txt_path = os.path.join(output_dir, os.path.splitext(name)[0] + ".txt")

        if os.path.exists(txt_path) and not args.overwrite:
            skipped += 1
            print(f"[{idx}/{total}] SKIP (exists): {name}")
            continue

        try:
            with Image.open(img_path) as im:
                im.load()
                result = model.caption(im, length="long")
            caption = result.get("caption", str(result)) if isinstance(result, dict) else str(result)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            done += 1
            print(f"[{idx}/{total}] OK: {name}")
        except Exception as e:
            failed += 1
            print(f"[{idx}/{total}] FAIL: {name} -> {e}")

    print(f"\nDone. Captioned: {done}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
