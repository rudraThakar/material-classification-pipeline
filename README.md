## End-to-End ML Pipeline for Real-Time Material Classification (Scrap Simulation)

### Overview
This repository implements a compact, end-to-end ML pipeline that simulates real-time scrap material classification using images. It covers dataset preparation, model training with transfer learning, lightweight deployment (TorchScript/ONNX), and a simulated real-time conveyor loop that classifies frames and logs results.

### Folder Structure
```
.
├── data/                   # Place dataset here or use a symlink; optional samples
├── models/                 # Saved checkpoints and exported models (TorchScript/ONNX)
├── results/                # Metrics, confusion matrix, logs, CSV outputs
├── src/
│   ├── data.py             # Dataset loading, transforms, splits
│   ├── train.py            # Transfer learning training script
│   ├── export.py           # Export to TorchScript and ONNX
│   ├── infer.py            # Single-image inference
│   └── simulate.py         # Simulated conveyor real-time loop
├── performance_report.md   # Short performance summary with visuals
├── CLEANUP.md              # What was downloaded/generated and how to delete
└── requirements.txt        # Dependencies
```

### Dataset
- Use any public dataset with ≥5 material classes (e.g., metal, plastic, glass, paper, cardboard/fabric/e-waste). Recommended options:
  - `TrashNet` (6 classes) — suitable for quick prototyping
  - `TACO` — more complex; requires parsing annotations
  - Any ImageFolder-style dataset you curate

Expected directory format for ImageFolder:
```
data/
  train/
    class_a/
      img1.jpg
      ...
    class_b/
    ...
  val/
    class_a/
    class_b/
    ...
```
If you only have a single `data/` folder with subfolders per class, `src/train.py` can auto-split into train/val.

### Quickstart
1) Create and activate environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Prepare data
- Place dataset under `data/` in ImageFolder format (see above). Optionally pass `--val_dir` if you have a separate validation folder; otherwise, the script will split from a single source.

3) Train model (ResNet18 transfer learning)
```bash
python src/train.py \
  --train_dir data/train \
  --val_dir data/val \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-3 \
  --img_size 224
```
Outputs:
- Best checkpoint: `models/best_model.pt`
- Class index mapping: `models/class_index.json`
- Confusion matrix and metrics: `results/`

4) Export to TorchScript and ONNX
```bash
python src/export.py \
  --checkpoint models/best_model.pt \
  --class_index models/class_index.json \
  --img_size 224
```
Outputs under `models/`: `model.torchscript.pt`, `model.onnx`

5) Single-image inference
```bash
python src/infer.py \
  --model models/model.torchscript.pt \
  --image path/to/image.jpg \
  --class_index models/class_index.json \
  --img_size 224
```

6) Simulated real-time conveyor loop
```bash
python src/simulate.py \
  --model models/model.torchscript.pt \
  --frames_dir data/val \
  --class_index models/class_index.json \
  --results_csv results/simulation_log.csv \
  --interval_ms 250 \
  --low_conf_threshold 0.55 \
  --img_size 224
```
Behavior:
- Iterates frames from a folder (or can be adapted to video capture)
- Prints predicted class and confidence
- Flags low-confidence predictions
- Appends results to a CSV in `results/`

### Notes
- The pipeline prioritizes simplicity and clarity. Swap the backbone (e.g., MobileNetV3) if you want a smaller model.
- ONNX export is provided for deployment scenarios; ensure opset compatibility with your target runtime.


