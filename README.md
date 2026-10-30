# PetVision — Cat & Dog Segmentation

Upload a pet photo → get a mask of the animal, or the same photo with the background removed.

A U-Net with an ImageNet-pretrained MobileNetV3 encoder, fine-tuned on the [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). The repo covers the whole ML workflow: data analysis → training → experiments → evaluation → optimization (ONNX / INT8) → web demo and REST API.

![demo](results/demo/demo_grid.png)

## Results at a glance

Official test set (3,669 images), evaluated at original image resolution:

| Model | mIoU | Dice | CPU latency (1 thread) | Size |
|---|---|---|---|---|
| PyTorch FP32 | 0.899 | 0.944 | 30.9 ms | 17.5 MB |
| ONNX FP32 (default in the app) | same | same | 13.7 ms | 17.3 MB |
| ONNX INT8 | −1.3 pts | −0.9 pts | 10.2 ms | 4.8 MB |

Key experiment findings (details in [REPORT.md](REPORT.md)):
- ImageNet pretraining vs training from scratch: **+14.5 IoU points**.
- Freezing the encoder matches full fine-tuning at **half the training cost**.
- 128 → 160 → 224 px: 0.891 → 0.900 → 0.910 test mIoU; 160 px chosen as the speed/accuracy compromise.
- ONNX export: **2.3–2.9× faster** with zero accuracy change.

## Project structure

```
petvision/
├── README.md                  ← you are here
├── REPORT.md                  ← technical report (problem, decisions, experiments, results)
├── requirements.txt
├── src/petvision/
│   ├── data.py                ← splits, trimap → binary mask, caching, augmentation
│   ├── model.py               ← U-Net + MobileNetV3, transfer-learning strategies
│   ├── metrics.py             ← IoU, Dice, pixel accuracy
│   ├── train.py               ← training CLI
│   ├── evaluate.py            ← test-set evaluation + error analysis figures
│   ├── optimize.py            ← ONNX export, INT8 quantization, benchmark
│   ├── inference.py           ← Segmenter class (PyTorch or ONNX backend), cutout/overlay helpers
│   └── predict.py             ← run on your own images
├── app/
│   ├── gradio_app.py          ← web demo (upload → mask / cutout)
│   ├── api.py                 ← FastAPI REST API
│   └── common.py              ← model loading
├── scripts/
│   ├── download_data.py       ← downloads the dataset into data/
│   ├── eda.py                 ← dataset analysis
│   ├── run_experiments.sh     ← experiments 1 & 2 (strategies, resolutions)
│   ├── run_final.sh           ← experiment 3 (15-epoch runs) + releases the final model
│   ├── compare_experiments.py ← evaluates all runs, makes tables + plots
│   ├── quant_ablation.py      ← INT8 calibration comparison
│   └── make_demo_figure.py    ← demo grid for README
├── weights/                   ← trained models (.pt, FP32 .onnx, INT8 .onnx)
├── runs/                      ← training logs of every experiment (history.json, summary.json)
├── results/                   ← metrics, tables and figures used in the report
├── assets/examples/           ← sample test images for the demo
└── tests/test_smoke.py        ← quick unit tests (no data needed)
```

## Setup

Python 3.10+.

```bash
git clone <your-repo-url> petvision && cd petvision
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# CPU-only PyTorch (smaller download). Skip this line if you have a CUDA GPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

pytest -q          # 4 quick tests, ~5 s
```

Main libraries: PyTorch, segmentation-models-pytorch, timm (pretrained MobileNetV3 from the Hugging Face hub), onnx + onnxruntime, FastAPI, Gradio, pandas, matplotlib.

## Quick start (use the trained model)

The trained weights are in `weights/`, so no training is needed.

**Web demo**
```bash
python app/gradio_app.py
# open http://localhost:7860, upload a cat/dog photo
```

**REST API**
```bash
uvicorn app.api:app --port 8000
# interactive docs: http://localhost:8000/docs

curl -F "file=@assets/examples/pug_197.jpg" "http://localhost:8000/segment?output=cutout" -o pug_cutout.png
curl -F "file=@assets/examples/pug_197.jpg" "http://localhost:8000/segment/info"
```
`output` can be `mask`, `cutout` (transparent PNG), `overlay`, or `background` (pet on a plain colour).

**Command line, on your own images**
```bash
PYTHONPATH=src python -m petvision.predict --model weights/petvision_unet_mnv3_fp32.onnx \
    --input path/to/photo_or_folder --out outputs/
# writes <name>_mask.png, <name>_overlay.jpg, <name>_cutout.png
```
Any of the three weight files works with `--model`. Add `--tta` for flip test-time augmentation (+0.25 IoU, ~1.6× slower). Set `PETVISION_MODEL=...` to choose the model for the demo/API.

> On Windows use `set PYTHONPATH=src` before the command (PowerShell: `$env:PYTHONPATH="src"`).

## Reproduce everything

```bash
export PYTHONPATH=src

# 1. data (~800 MB)
python scripts/download_data.py

# 2. EDA → results/eda/
python scripts/eda.py

# 3. experiments: 5 transfer strategies @128 px, then resolutions 160/224 px (6 epochs each)
bash scripts/run_experiments.sh          # ~1.5 h on a 2-core CPU, minutes on a GPU

# 4. 15-epoch frozen vs fine-tune @160 px; copies the winner to weights/petvision_unet_mnv3.pt
bash scripts/run_final.sh                # ~65 min on CPU

# 5. compare all runs on the test set → results/experiments/
python scripts/compare_experiments.py

# 6. full test evaluation + error analysis → results/eval_final/
python -m petvision.evaluate --model weights/petvision_unet_mnv3.pt --out results/eval_final

# 7. ONNX + INT8, speed/size/accuracy comparison → weights/, results/optimization/
python -m petvision.optimize --model weights/petvision_unet_mnv3.pt --out-dir weights
python scripts/quant_ablation.py
```

Train a single model with your own settings:
```bash
python -m petvision.train --strategy frozen --size 224 --epochs 20 --out runs/my_run
# strategies: scratch | frozen | finetune | discrim | two_stage
```
The first run at a given `--size` builds a resized cache in `data/cache/` (a minute or two). Training uses a CUDA or Apple MPS GPU automatically if available.

## Getting the trained model

The weights are committed in `weights/` (largest file 18 MB, under GitHub's 100 MB limit). If you'd rather not keep binaries in git, upload the `weights/` files to a GitHub Release, or recreate them with steps 1, 4 and 7 above (the final model needs only `run_final.sh` → the frozen run takes ~23 min on CPU).

## Notes

- Dataset: Oxford-IIIT Pet, Parkhi et al. 2012, released under CC BY-SA 4.0. The few images in `assets/examples/` come from its test split.
- Labels: trimap "pet" and "boundary" are both treated as foreground (see REPORT.md, section 2).
- All reported numbers come from CPU runs in this repo; logs are in `runs/` and `results/`.
