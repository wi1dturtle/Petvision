"""Web demo: upload a pet photo -> mask, overlay, transparent cutout, new background.

Run:  python app/gradio_app.py      then open http://localhost:7860
"""
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.common import ROOT, load_segmenter  # noqa: E402
from petvision.inference import cutout, mask_image, overlay, replace_background  # noqa: E402

SEG = load_segmenter()
EXAMPLES = sorted(str(p) for p in (ROOT / "assets" / "examples").glob("*.jpg"))


def parse_color(value: str, default=(30, 144, 255)):
    """Accepts '#rrggbb' or 'rgb(a)(r, g, b, ...)' (different Gradio versions return either)."""
    value = (value or "").strip()
    try:
        if value.startswith("#"):
            return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
        if value.startswith("rgb"):
            parts = value[value.index("(") + 1:value.index(")")].split(",")
            return tuple(int(float(p)) for p in parts[:3])
    except ValueError:
        pass
    return default


def run(img: Image.Image, threshold: float, bg_hex: str):
    if img is None:
        return None, None, None, None, "Upload an image first."
    img = img.convert("RGB")
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600))
    t0 = time.perf_counter()
    prob = SEG.predict_proba([img])[0]
    ms = (time.perf_counter() - t0) * 1000
    mask = (prob > threshold).astype(np.uint8)
    bg = parse_color(bg_hex)
    info = (f"Pet covers {mask.mean() * 100:.1f}% of the image · model time {ms:.0f} ms "
            f"({SEG.backend}, {SEG.size}px input)")
    if mask.mean() < 0.01:
        info += " · no pet found (the model is trained on cats and dogs only)"
    return mask_image(mask), overlay(img, mask), cutout(img, prob), replace_background(img, prob, bg), info


with gr.Blocks(title="PetVision") as demo:
    gr.Markdown("# PetVision\nUpload a photo of a cat or dog. The model separates the animal from the background.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Pet photo")
            thr = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Mask threshold")
            bg = gr.ColorPicker(value="#1e90ff", label="New background colour")
            btn = gr.Button("Segment", variant="primary")
            info = gr.Markdown()
        with gr.Column():
            with gr.Row():
                o_mask = gr.Image(label="Mask")
                o_overlay = gr.Image(label="Overlay")
            with gr.Row():
                o_cut = gr.Image(label="Background removed (PNG)", format="png", image_mode="RGBA")
                o_bg = gr.Image(label="New background")
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=inp)
    btn.click(run, [inp, thr, bg], [o_mask, o_overlay, o_cut, o_bg, info])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
