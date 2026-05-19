"""
Moondream3 Gradio UI
Web interface for the Moondream3 vision-language model.
"""

import os
import sys
import warnings

os.environ["TORCH_COMPILE_DISABLE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Force UTF-8 stdout/stderr so Unicode chars (✓ ⚠️ 🚀) don't crash the app
# on Windows shells using cp1252 (e.g. inside Pinokio's bundled terminal).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
print(f"PyTorch version: {torch.__version__}")

# Monkey-patch BlockMask to add missing seq_lengths attribute
try:
    from torch.nn.attention.flex_attention import BlockMask
    original_init = BlockMask.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'seq_lengths'):
            if hasattr(self, 'shape'):
                self.seq_lengths = self.shape
            elif hasattr(self, 'kv_num_blocks'):
                self.seq_lengths = (self.kv_num_blocks[-1] * 128, self.kv_num_blocks[-1] * 128)
            else:
                self.seq_lengths = None

    BlockMask.__init__ = patched_init
    print("✓ Applied BlockMask patch for seq_lengths compatibility")
except Exception as e:
    print(f"⚠️ Could not patch BlockMask: {e}")

import gradio as gr
from transformers import AutoModelForCausalLM
from PIL import Image, ImageDraw

model = None


def load_model():
    """Load the Moondream3 model."""
    global model
    if model is not None:
        return "Model already loaded!"

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    try:
        print(f"Loading Moondream3 on {device}...")

        model = AutoModelForCausalLM.from_pretrained(
            "moondream/moondream3-preview",
            trust_remote_code=True,
            dtype=dtype,
            device_map={"": device},
        )

        print("✓ Model loaded successfully!")

        try:
            print("Compiling model for optimized inference...")
            model.compile()
            print("✓ Model compiled successfully!")
            return f"Model loaded and compiled successfully on {device}!\n\nRunning with full optimization."
        except Exception as compile_error:
            print(f"⚠️ Compilation failed: {compile_error}")
            return f"Model loaded on {device}!\n\n⚠️ Compilation skipped: {str(compile_error)[:100]}\n\nModel will work but may be slower."

    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        return f"Error loading model: {error_msg}"


def check_model():
    """Check if the model is loaded."""
    if model is None:
        return False, "Please load the model first by clicking 'Load Model'!"
    return True, None


def build_settings(temperature, max_tokens):
    """Build settings dict from UI values."""
    settings = {}
    if temperature is not None and temperature > 0:
        settings["temperature"] = temperature
    if max_tokens is not None and max_tokens > 0:
        settings["max_tokens"] = int(max_tokens)
    return settings if settings else None


def caption_image(image, length, temperature, max_tokens, stream):
    """Generate a caption for the image."""
    loaded, error = check_model()
    if not loaded:
        yield error
        return

    if image is None:
        yield "Please upload an image!"
        return

    try:
        settings = build_settings(temperature, max_tokens)
        kwargs = {"length": length}
        if settings:
            kwargs["settings"] = settings

        if stream:
            kwargs["stream"] = True
            result = model.caption(image, **kwargs)
            caption_stream = result.get("caption", result) if isinstance(result, dict) else result
            text = ""
            for chunk in caption_stream:
                text += chunk
                yield text
        else:
            result = model.caption(image, **kwargs)
            if isinstance(result, dict):
                yield result.get("caption", str(result))
            else:
                yield str(result)
    except Exception as e:
        yield f"Error: {e}"


def answer_question(image, question, reasoning, temperature, max_tokens, stream):
    """Answer a question about the image (or text-only)."""
    loaded, error = check_model()
    if not loaded:
        yield error
        return

    if not question or question.strip() == "":
        yield "Please enter a question!"
        return

    try:
        settings = build_settings(temperature, max_tokens)
        kwargs = {"question": question, "reasoning": reasoning}
        if image is not None:
            kwargs["image"] = image
        if settings:
            kwargs["settings"] = settings

        if stream:
            kwargs["stream"] = True
            result = model.query(**kwargs)
            answer_stream = result.get("answer", result) if isinstance(result, dict) else result
            text = ""
            for chunk in answer_stream:
                text += chunk
                yield text
        else:
            result = model.query(**kwargs)
            if isinstance(result, dict):
                yield result.get("answer", str(result))
            else:
                yield str(result)
    except Exception as e:
        yield f"Error: {e}"


def detect_objects(image, object_type, max_objects):
    """Detect objects in the image."""
    loaded, error = check_model()
    if not loaded:
        return None, error

    if image is None:
        return None, "Please upload an image!"

    if not object_type or object_type.strip() == "":
        return None, "Please specify an object type!"

    try:
        kwargs = {}
        if max_objects is not None and max_objects > 0:
            kwargs["settings"] = {"max_objects": int(max_objects)}

        result = model.detect(image, object_type.strip(), **kwargs)

        if isinstance(result, dict):
            objects = result.get("objects", [])
        else:
            return image, f"Unexpected result: {str(result)}"

        if not objects:
            return image, f"No '{object_type}' detected."

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size

        for obj in objects:
            x_min = int(obj.get("x_min", 0) * width)
            y_min = int(obj.get("y_min", 0) * height)
            x_max = int(obj.get("x_max", 0) * width)
            y_max = int(obj.get("y_max", 0) * height)

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            label = obj.get("label", object_type)
            draw.text((x_min, max(0, y_min - 20)), label, fill="red")

        return annotated, f"✓ Detected {len(objects)} object(s)."
    except Exception as e:
        return image, f"Error: {e}"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def _collect_batch_inputs(folder_path, uploaded_files):
    """Return a list of (source_image_path, base_name) pairs from folder + uploads."""
    items = []
    seen = set()

    if folder_path and folder_path.strip():
        folder = os.path.abspath(folder_path.strip().strip('"').strip("'"))
        if not os.path.isdir(folder):
            return None, f"Folder not found: {folder}"
        for f in sorted(os.listdir(folder)):
            full = os.path.join(folder, f)
            if (
                os.path.splitext(f)[1].lower() in IMAGE_EXTS
                and os.path.isfile(full)
            ):
                key = os.path.normcase(full)
                if key not in seen:
                    seen.add(key)
                    items.append((full, os.path.splitext(f)[0]))

    if uploaded_files:
        for fobj in uploaded_files:
            full = fobj if isinstance(fobj, str) else getattr(fobj, "name", None)
            if not full or not os.path.isfile(full):
                continue
            if os.path.splitext(full)[1].lower() not in IMAGE_EXTS:
                continue
            key = os.path.normcase(os.path.abspath(full))
            if key not in seen:
                seen.add(key)
                items.append((full, os.path.splitext(os.path.basename(full))[0]))

    return items, None


def batch_caption_folder(
    folder_path,
    uploaded_files,
    output_folder,
    overwrite,
    length,
    progress=gr.Progress(),
):
    """Caption images from a folder path and/or uploaded files."""
    loaded, error = check_model()
    if not loaded:
        yield error
        return

    items, err = _collect_batch_inputs(folder_path, uploaded_files)
    if err:
        yield err
        return

    if not items:
        yield "No images found. Provide a folder path or upload image files."
        return

    out_dir = None
    if output_folder and output_folder.strip():
        out_dir = os.path.abspath(output_folder.strip().strip('"').strip("'"))
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            yield f"Cannot create output folder '{out_dir}': {e}"
            return

    total = len(items)
    header = f"Captioning {total} image(s) — length: {length}\n"
    if out_dir:
        header += f"Output folder: {out_dir}\n"
    else:
        header += "Output: sidecar .txt next to each source image\n"
    log_lines = [header]
    yield "\n".join(log_lines)

    done = skipped = failed = 0

    for idx, (img_path, base) in enumerate(items, start=1):
        progress(idx / total, desc=f"{idx}/{total} {os.path.basename(img_path)}")
        txt_dir = out_dir if out_dir else os.path.dirname(img_path)
        txt_path = os.path.join(txt_dir, base + ".txt")

        if os.path.exists(txt_path) and not overwrite:
            skipped += 1
            log_lines.append(f"[{idx}/{total}] SKIP (exists): {base}.txt")
            yield "\n".join(log_lines)
            continue

        try:
            with Image.open(img_path) as im:
                im.load()
                result = model.caption(im, length=length)
            caption = result.get("caption", str(result)) if isinstance(result, dict) else str(result)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            done += 1
            log_lines.append(f"[{idx}/{total}] OK: {os.path.basename(img_path)}")
        except Exception as e:
            failed += 1
            log_lines.append(f"[{idx}/{total}] FAIL: {os.path.basename(img_path)} -> {e}")

        yield "\n".join(log_lines)

    log_lines.append(f"\nDone. Captioned: {done}, Skipped: {skipped}, Failed: {failed}")
    yield "\n".join(log_lines)


def _hf_cache_root():
    """Return the active HuggingFace hub cache directory."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.path.abspath(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return os.path.abspath(os.path.join(os.environ["HF_HOME"], "hub"))
    return os.path.abspath(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))


def _dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _format_size(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def hf_cache_status():
    """Report current HF cache state, model presence and offline mode."""
    cache_root = _hf_cache_root()
    offline = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
    model_dir = os.path.join(cache_root, "models--moondream--moondream3-preview")
    has_model = os.path.isdir(model_dir)

    lines = [
        f"HF cache root : {cache_root}",
        f"HF_HOME       : {os.environ.get('HF_HOME', '(not set)')}",
        f"HF_HUB_CACHE  : {os.environ.get('HF_HUB_CACHE', '(not set)')}",
        f"Offline mode  : {'ON' if offline else 'OFF'} (HF_HUB_OFFLINE)",
        f"Moondream3    : {'cached locally' if has_model else 'NOT cached'} ({model_dir})",
    ]
    if has_model:
        try:
            lines.append(f"Model size    : {_format_size(_dir_size_bytes(model_dir))}")
        except Exception as e:
            lines.append(f"Model size    : (error: {e})")
    if os.path.isdir(cache_root):
        try:
            lines.append(f"Cache size    : {_format_size(_dir_size_bytes(cache_root))}")
        except Exception as e:
            lines.append(f"Cache size    : (error: {e})")
    return "\n".join(lines)


def hf_set_offline(enabled):
    """Toggle HF offline mode. Takes effect on next model load."""
    os.environ["HF_HUB_OFFLINE"] = "1" if enabled else "0"
    return hf_cache_status()


def hf_set_cache_path(new_path):
    """Override HF cache location. Only affects future loads in this process."""
    if not new_path or not new_path.strip():
        return hf_cache_status() + "\n\n(no change — provide a path)"
    new_path = os.path.abspath(new_path.strip().strip('"').strip("'"))
    try:
        os.makedirs(new_path, exist_ok=True)
    except Exception as e:
        return hf_cache_status() + f"\n\nError creating '{new_path}': {e}"
    os.environ["HF_HOME"] = new_path
    os.environ["HF_HUB_CACHE"] = os.path.join(new_path, "hub")
    return hf_cache_status() + "\n\n✓ Updated. Restart the app for it to apply to the loaded model."


def hf_open_cache():
    """Open the HF cache folder in the system file explorer."""
    cache_root = _hf_cache_root()
    if not os.path.isdir(cache_root):
        return f"Cache folder does not exist yet: {cache_root}"
    try:
        import subprocess, sys
        if sys.platform == "win32":
            os.startfile(cache_root)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", cache_root])
        else:
            subprocess.Popen(["xdg-open", cache_root])
        return f"Opened: {cache_root}"
    except Exception as e:
        return f"Could not open '{cache_root}': {e}"


def point_objects(image, object_type):
    """Point to objects in the image."""
    loaded, error = check_model()
    if not loaded:
        return None, error

    if image is None:
        return None, "Please upload an image!"

    if not object_type or object_type.strip() == "":
        return None, "Please specify an object type!"

    try:
        result = model.point(image, object_type.strip())

        if isinstance(result, dict):
            points = result.get("points", [])
        else:
            return image, f"Unexpected result: {str(result)}"

        if not points:
            return image, f"No '{object_type}' found."

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size

        for point in points:
            x = int(point.get("x", 0) * width)
            y = int(point.get("y", 0) * height)

            radius = 20
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                        outline="#0066CC", width=4)
            inner_radius = 16
            draw.ellipse([x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius],
                        fill="#4DA6FF", outline="#0066CC", width=2)

        return annotated, f"✓ Found {len(points)} point(s)."
    except Exception as e:
        return image, f"Error: {e}"


# Gradio interface
with gr.Blocks(title="Moondream3 Vision AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌙 Moondream3 Vision AI

        Vision-language model with image captioning, visual Q&A, object detection, and pointing.

        **Click "Load Model" to start.**
        """
    )

    with gr.Row():
        load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
        load_status = gr.Textbox(label="Status", value="Model not loaded", interactive=False, scale=3, lines=3)

    load_btn.click(fn=load_model, outputs=load_status)

    gr.Markdown("---")

    with gr.Tabs():
        with gr.TabItem("📝 Image Captioning"):
            with gr.Row():
                with gr.Column():
                    caption_image_input = gr.Image(type="pil", label="Upload Image")
                    caption_length = gr.Radio(choices=["short", "normal", "long"], value="normal", label="Length")
                    with gr.Accordion("Advanced Settings", open=False):
                        caption_temperature = gr.Slider(0, 1.5, value=0, step=0.1, label="Temperature (0 = default)")
                        caption_max_tokens = gr.Slider(0, 1024, value=0, step=64, label="Max Tokens (0 = default)")
                        caption_stream = gr.Checkbox(label="Stream Output", value=False)
                    caption_btn = gr.Button("Generate Caption", variant="primary")
                with gr.Column():
                    caption_output = gr.Textbox(label="Caption", lines=5)

            caption_btn.click(
                fn=caption_image,
                inputs=[caption_image_input, caption_length, caption_temperature, caption_max_tokens, caption_stream],
                outputs=caption_output,
            )

        with gr.TabItem("❓ Visual Q&A"):
            with gr.Row():
                with gr.Column():
                    vqa_image_input = gr.Image(type="pil", label="Upload Image (optional for text-only)")
                    vqa_question = gr.Textbox(label="Question", placeholder="What is in this image?", lines=2)
                    vqa_reasoning = gr.Checkbox(label="Enable Reasoning", value=True)
                    with gr.Accordion("Advanced Settings", open=False):
                        vqa_temperature = gr.Slider(0, 1.5, value=0, step=0.1, label="Temperature (0 = default)")
                        vqa_max_tokens = gr.Slider(0, 2048, value=0, step=64, label="Max Tokens (0 = default)")
                        vqa_stream = gr.Checkbox(label="Stream Output", value=False)
                    vqa_btn = gr.Button("Ask Question", variant="primary")
                with gr.Column():
                    vqa_output = gr.Textbox(label="Answer", lines=8)

            vqa_btn.click(
                fn=answer_question,
                inputs=[vqa_image_input, vqa_question, vqa_reasoning, vqa_temperature, vqa_max_tokens, vqa_stream],
                outputs=vqa_output,
            )

        with gr.TabItem("🔍 Object Detection"):
            with gr.Row():
                with gr.Column():
                    detect_image_input = gr.Image(type="pil", label="Upload Image")
                    detect_object_type = gr.Textbox(label="Object", placeholder="person, car, dog", lines=1)
                    detect_max_objects = gr.Slider(0, 50, value=0, step=1, label="Max Objects (0 = default)")
                    detect_btn = gr.Button("Detect", variant="primary")
                with gr.Column():
                    detect_image_output = gr.Image(type="pil", label="Result")
                    detect_text_output = gr.Textbox(label="Info", lines=3)

            detect_btn.click(
                fn=detect_objects,
                inputs=[detect_image_input, detect_object_type, detect_max_objects],
                outputs=[detect_image_output, detect_text_output],
            )

        with gr.TabItem("👆 Object Pointing"):
            with gr.Row():
                with gr.Column():
                    point_image_input = gr.Image(type="pil", label="Upload Image")
                    point_object_type = gr.Textbox(label="Object", placeholder="person, car, dog", lines=1)
                    point_btn = gr.Button("Point", variant="primary")
                with gr.Column():
                    point_image_output = gr.Image(type="pil", label="Result")
                    point_text_output = gr.Textbox(label="Info", lines=3)

            point_btn.click(
                fn=point_objects,
                inputs=[point_image_input, point_object_type],
                outputs=[point_image_output, point_text_output],
            )

        with gr.TabItem("📁 Batch Caption"):
            gr.Markdown(
                "Caption many images at once. Use **either** a server-side folder path, "
                "**or** drag-and-drop files, **or both** — duplicates are filtered."
            )
            with gr.Row():
                with gr.Column():
                    batch_folder = gr.Textbox(
                        label="Folder Path (server-side)",
                        placeholder=r"e.g. D:\photos\my_album  (leave empty if only uploading files)",
                        lines=1,
                    )
                    batch_files = gr.File(
                        label="Or upload images (drag & drop, multi-select)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    batch_output_folder = gr.Textbox(
                        label="Output Folder (optional)",
                        placeholder="Leave empty to write a .txt next to each source image",
                        lines=1,
                    )
                    batch_length = gr.Radio(
                        choices=["short", "normal", "long"],
                        value="long",
                        label="Caption Length",
                    )
                    batch_overwrite = gr.Checkbox(label="Overwrite existing .txt files", value=False)
                    batch_btn = gr.Button("Caption All Images", variant="primary")
                with gr.Column():
                    batch_output = gr.Textbox(label="Progress", lines=22)

            batch_btn.click(
                fn=batch_caption_folder,
                inputs=[
                    batch_folder,
                    batch_files,
                    batch_output_folder,
                    batch_overwrite,
                    batch_length,
                ],
                outputs=batch_output,
            )

        with gr.TabItem("💾 HF Cache"):
            gr.Markdown(
                "Manage the HuggingFace cache used for the Moondream3 weights. "
                "Path/offline changes only take effect on the **next model load** "
                "(restart the app)."
            )
            with gr.Row():
                with gr.Column():
                    hf_status = gr.Textbox(
                        label="Cache Status",
                        value=hf_cache_status(),
                        lines=10,
                        interactive=False,
                    )
                    hf_refresh_btn = gr.Button("🔄 Refresh status")
                    hf_open_btn = gr.Button("📂 Open cache folder")
                with gr.Column():
                    hf_offline_toggle = gr.Checkbox(
                        label="Offline mode (HF_HUB_OFFLINE)",
                        value=os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True"),
                    )
                    hf_cache_path_input = gr.Textbox(
                        label="Set custom HF cache path (HF_HOME)",
                        placeholder=r"e.g. D:\hf_cache  (leave empty to keep default)",
                        lines=1,
                    )
                    hf_apply_path_btn = gr.Button("Apply cache path", variant="primary")

            hf_refresh_btn.click(fn=hf_cache_status, outputs=hf_status)
            hf_open_btn.click(fn=hf_open_cache, outputs=hf_status)
            hf_offline_toggle.change(fn=hf_set_offline, inputs=hf_offline_toggle, outputs=hf_status)
            hf_apply_path_btn.click(fn=hf_set_cache_path, inputs=hf_cache_path_input, outputs=hf_status)

    gr.Markdown(
        """
        ---
        *Powered by [Moondream3](https://huggingface.co/moondream/moondream3-preview) & [Gradio](https://gradio.app)*
        """
    )


if __name__ == "__main__":
    print("Starting Moondream3...")
    demo.launch(share=False)
