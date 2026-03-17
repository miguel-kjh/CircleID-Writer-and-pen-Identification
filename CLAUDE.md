# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition: [ICDAR 2026 CircleID Writer Identification](https://www.kaggle.com/competitions/icdar-2026-circleid-writer-identification/data). The goal is to identify the **writer** and/or **pen** used in handwriting samples (circle drawings).

Two tasks:
- **Writer Identification**: Classify which writer (W01–W51) wrote the sample, with an "unknown writer" option (`-1`) via confidence threshold.
- **Pen Classification**: Classify which pen (1–8) was used.

## Commands

```bash
# Training
python train.py                                      # writer task, resnet18, defaults
python train.py --task pen
python train.py --task writer --model resnet18 --epochs 20 --batch-size 64 --lr 1e-4

# Inference (requires a prior training run)
python predict.py                                    # writer task, resnet18
python predict.py --task pen --model resnet18

# Notebook (legacy baseline)
pip install numpy pandas pillow torch torchvision tqdm
jupyter notebook baseline.ipynb
```

## Data Layout

```
dataset/
  raw/
    train.csv              # image_id, image_path, writer_id, pen_id
    additional_train.csv   # same schema; writer_id=-1 means unknown writer
    test.csv               # image_id, image_path (no labels)
    sample_submission.csv  # image_id, writer_id (example format)
  images/                  # PNG handwriting samples
results/                   # run directories with checkpoints, logs, submission CSVs
```

Train images use numeric filenames (`00001.png`). Test images use hex-string filenames (`v2_<hash>.png`). `image_path` in each CSV is relative to `dataset/` (the `IMAGE_DIR` in `Config`), while CSVs themselves live in `dataset/raw/`.

## Architecture

The codebase is a modular refactor of `baseline.ipynb` into `train.py` / `predict.py` backed by `src/`.

### Configuration (`src/config.py`)

`Config` is a class with class-level attributes and dynamic `@property` methods:
- `run_dir` generates a deterministic path like `resnet18_writer_e10_bs128_lr3e-4_img224_seed0/` under `OUTPUT_DIR`
- `best_ckpt_path` / `ckpt_path` / `log_path` all resolve under `run_dir`
- `setup()` creates the output directories

`train.py` and `predict.py` parse CLI args and mutate a `Config` instance before use.

### Model Registry (`src/models/`)

Models follow a registry pattern:
- `BaseModel(nn.Module)` requires a `NAME` class attribute
- `_REGISTRY` in `src/models/__init__.py` maps name → class
- `build_model(name, num_classes)` is the factory used by both scripts

To add a new model: subclass `BaseModel`, set `NAME`, add to `_REGISTRY`. No changes needed in `train.py`/`predict.py`.

### Training Loop (`src/models/train.py`)

`train_epoch`, `evaluate`, and `predict` are standalone functions (not methods). `predict` applies softmax confidence thresholding for the writer task (below `WRITER_UNKNOWN_THRESHOLD` → predicts `-1`); pen task uses argmax.

### Data (`src/data/`)

- `CircleDataset` resolves image paths as `img_root / image_path` from the CSV
- ImageNet normalization from `ResNet18_Weights.DEFAULT`; train split uses random rotation ±10° augmentation
- `random_split` is **not** stratified by writer — some writers may be absent from validation

### Experiment Tracking

wandb is integrated in `train.py` (project: `"circleid"`). Run name mirrors `run_dir`. Metrics logged: `train/loss`, `val/loss`, `val/acc` per epoch, plus `best_val_acc` as a summary.

## Key Design Decisions

- **Unknown writer**: Samples with softmax confidence < `WRITER_UNKNOWN_THRESHOLD` (default 0.9) are predicted as `-1`. The threshold is the primary heuristic to tune.
- **`additional_train.csv`** contains samples with `writer_id=-1` (genuinely unknown writers) — not used in the current baseline but relevant for training an unknown-writer detector.
- Label maps (`label_map` str→int, `idx_map` int→str) are built from `train.csv` only and persisted in `log.json` so inference uses the same indices as training.
- Inference always loads `best_ckpt_path` (best val acc checkpoint), not the last checkpoint.

## Baseline Results

Writer task (10 epochs, ResNet18, `lr=3e-4`, `batch_size=128`): **~92% validation accuracy** on a random 80/20 split of `train.csv`.
