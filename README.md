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


### Training Pipeline Details
- Data ingestion
  - ImageFolder structure (`data/train`, `data/val`).
  - Class index mapping saved to `models/class_index.json`.
- Preprocessing and augmentations (train)
  - RandomResizedCrop to 224×224 (scale 0.7–1.0, slight aspect jitter).
  - RandomHorizontalFlip (p=0.5).
  - ColorJitter (brightness/contrast/saturation/hue small ranges).
  - AutoAugment (ImageNet policy) when available.
  - ToTensor + Normalize (ImageNet mean/std).
  - RandomErasing (p=0.1).
- Preprocessing (validation)
  - Resize to 224×224, ToTensor, Normalize (ImageNet mean/std).
- Model
  - Backbone: `torchvision.models.resnet18` with pretrained weights (optional via `--no_pretrain`).
  - Replace final FC with `Linear(in_features, num_classes)`.
- Optimization
  - Optimizer: AdamW, lr=1e-3 (configurable), weight_decay=1e-4.
  - Loss: CrossEntropyLoss with label smoothing (default 0.05).
  - Warmup: linear warmup for first `--warmup_epochs`.
  - Scheduler: CosineAnnealingLR for remaining epochs.
- Fine-tuning schedule
  - Freeze all layers except FC for `--freeze_epochs` (default 1).
  - Unfreeze whole network afterwards and continue training.
- Training loop
  - Mixed CPU/GPU support; `--cpu` forces CPU.
  - Per-epoch: train loss tracking; evaluate on val set each epoch.
- Metrics and evaluation
  - Metrics: accuracy, macro precision, macro recall.
  - Confusion matrix computed each epoch; CSV written to `results/`.
  - Training history written to `results/training_history.csv`.
- Checkpointing
  - Best validation accuracy checkpoint saved to `models/best_model.pt` with `model_state`, `img_size`, and `num_classes`.
- Export (deployment)
  - TorchScript tracing to `models/model.torchscript.pt`.
  - ONNX export to `models/model.onnx` (dynamic batch; opset 17).
- Inference
  - TorchScript via PyTorch or ONNX via ONNX Runtime.
  - Outputs predicted class and confidence; low-confidence flag based on threshold.
- Simulation loop
  - Iterates images from a folder at intervals, prints prediction and logs to CSV.

