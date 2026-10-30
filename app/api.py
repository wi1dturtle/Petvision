"""REST API for pet segmentation.

Run:   uvicorn app.api:app --port 8000
Docs:  http://localhost:8000/docs

    curl -F "file=@cat.jpg" "http://localhost:8000/segment?output=cutout" -o cat_cutout.png
    curl -F "file=@cat.jpg" "http://localhost:8000/segment/info"
"""
import io
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from app.common import load_segmenter
from petvision.inference import cutout, mask_image, overlay, replace_background

app = FastAPI(title="PetVision API", version="1.0.0",
              description="Cat/dog foreground segmentation (U-Net + MobileNetV3, Oxford-IIIT Pet).")
SEG = load_segmenter()
MAX_SIDE = 2048


def _read(file: UploadFile) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "could not read the uploaded file as an image")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im


def _png(img: Image.Image, fmt="PNG") -> Response:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return Response(buf.getvalue(), media_type=f"image/{fmt.lower()}")


@app.get("/health")
def health():
    return {"status": "ok", "backend": SEG.backend, "input_size": SEG.size}


@app.post("/segment")
def segment(file: UploadFile = File(...),
            output: str = Query("cutout", pattern="^(mask|cutout|overlay|background)$"),
            threshold: float = Query(0.5, ge=0.0, le=1.0)):
    """Returns an image: binary mask, RGBA cutout, colored overlay, or the pet on a plain background."""
    im = _read(file)
    prob = SEG.predict_proba([im])[0]
    mask = (prob > threshold).astype(np.uint8)
    if output == "mask":
        return _png(mask_image(mask))
    if output == "overlay":
        return _png(overlay(im, mask), "JPEG")
    if output == "background":
        return _png(replace_background(im, prob), "JPEG")
    return _png(cutout(im, prob))


@app.post("/segment/info")
def segment_info(file: UploadFile = File(...), threshold: float = 0.5):
    """JSON summary: how much of the image is the pet, its bounding box, and model latency."""
    im = _read(file)
    t0 = time.perf_counter()
    prob = SEG.predict_proba([im])[0]
    ms = (time.perf_counter() - t0) * 1000
    mask = prob > threshold
    ys, xs = np.nonzero(mask)
    bbox = None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {"width": im.size[0], "height": im.size[1], "foreground_fraction": float(mask.mean()),
            "pet_detected": bool(mask.mean() > 0.01), "bbox_xyxy": bbox,
            "mean_confidence_in_mask": float(prob[mask].mean()) if mask.any() else 0.0,
            "inference_ms": round(ms, 1), "backend": SEG.backend}
