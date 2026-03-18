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
python train.py                                                        # writer task, resnet18, defaults
python train.py --task pen
python train.py --task writer --model resnet18 --epochs 20 --batch-size 64 --lr 1e-4
python train.py --dataset raw_join                                     # use merged train+additional_train

# Inference — must pass the same run-identifying flags as training to resolve run_dir
python predict.py --task writer --epochs 20 --batch-size 64 --lr 1e-4
python predict.py --task pen --model resnet18
python predict.py --dataset raw_join                                   # must match training flag

# Notebook (legacy baseline)
pip install numpy pandas pillow torch torchvision tqdm pytorch-lightning
jupyter notebook baseline.ipynb
```

`predict.py` accepts the same flags as `train.py` (including `--epochs`, `--lr`, `--seed`, `--val-frac`) because `run_dir` — and therefore the checkpoint path — is derived from all of them via `Config.run_dir`.

## Data Layout

```
dataset/
  raw/
    train.csv              # image_id, image_path, writer_id, pen_id  (23 850 rows)
    additional_train.csv   # same schema; writer_id=-1 means unknown writer (16 400 rows)
    test.csv               # image_id, image_path (no labels)
    sample_submission.csv  # image_id, writer_id (example format)
  raw_join/
    train.csv              # union of raw/train.csv + raw/additional_train.csv (40 250 rows)
    test.csv               # symlink → raw/test.csv
  images/                  # PNG handwriting samples
results/                   # run directories: checkpoints, log.json, submission CSVs
```

Select a dataset with `--dataset raw` (default) or `--dataset raw_join`. The `run_dir` encodes `_ds{dataset}` when the dataset is not `raw`, so runs with different datasets never collide.

Train images use numeric filenames (`00001.png`). Test images use hex-string filenames (`v2_<hash>.png`). `image_path` in each CSV is relative to `dataset/` (`IMAGE_DIR` in `Config`); CSVs live in `dataset/raw/`.

## Architecture

The codebase uses **PyTorch Lightning** for training and inference, backed by `src/`.

### Configuration (`src/config.py`)

`Config` is a plain class with mutable attributes. Its `@property` methods compute paths:
- `run_dir` → `{OUTPUT_DIR}/{model}_{task}_e{E}_bs{BS}_lr{lr}_img{S}_seed{seed}/`
- `best_ckpt_path`, `ckpt_path`, `log_path` all resolve under `run_dir`
- `setup()` creates `run_dir` on disk

Both scripts call `parse_args()` which mutates a `Config` instance then calls `cfg.setup()`.

### Model Registry (`src/models/`)

- `BaseModel(nn.Module)` — abstract base requiring a `NAME` class attribute
- `_REGISTRY` in `src/models/__init__.py` maps `name → class`
- `build_model(name, num_classes)` is the factory used everywhere

To add a new model: subclass `BaseModel`, set `NAME`, import and add to `_REGISTRY`. No changes needed in `train.py`/`predict.py`.

### Lightning Module (`src/models/lightning_module.py`)

`CircleIDModule(pl.LightningModule)` wraps any `BaseModel`:
- Constructor takes `net`, `lr`, `task`, `idx_map`, `writer_unknown_threshold`
- `idx_map` is stored with **string keys** (`{"0": "W01", ...}`) via `save_hyperparameters` for YAML round-trip safety
- `predict_step` returns `List[(image_id, label)]` per batch; applies softmax thresholding for the writer task
- `from_checkpoint(ckpt_path, net_builder)` classmethod rebuilds the net from hparams and loads weights

### Data (`src/data/`)

- `CircleDataset` — resolves `img_root / image_path`; returns `(tensor, label)` or `(tensor, image_id)` depending on `return_label`; ImageNet normalisation from `ResNet18_Weights.DEFAULT`; train split gets random rotation ±10°
- `CircleDataModule(pl.LightningDataModule)` — `setup("fit")` reads `train.csv`, builds label maps, splits, writes `log.json`; `setup("predict")` reads `test.csv`
- `random_split` is **not** stratified — some writers may be absent from validation
- Label maps (`label_map` str→int, `idx_map` int→str) are built from `train.csv` only and written to `log.json` so inference uses identical indices

### Training Flow (`train.py`)

1. `dm.setup("fit")` populates `label_map`/`idx_map` before the model is built
2. `CircleIDModule` is constructed with those maps
3. `WandbLogger` (project `"circleid"`, run name = `run_dir` basename) + two `ModelCheckpoint` callbacks (`checkpoint_best.ckpt` monitors `val/acc`, `checkpoint.ckpt` saves every epoch)
4. `trainer.fit(module, datamodule=dm)`
5. `log.json` is updated with `best_ckpt_path` (Lightning `.ckpt` path)
6. `trainer.predict(...)` generates the submission CSV inside `run_dir`

### Inference Flow (`predict.py`)

1. Reads `log.json` to get `best_ckpt_path`
2. `CircleIDModule.from_checkpoint(ckpt_path, net_builder)` restores full module from the Lightning checkpoint
3. `trainer.predict(module, datamodule=dm, ckpt_path=ckpt_path)` generates submission CSV in `OUTPUT_DIR`

## Key Design Decisions

- **Unknown writer threshold**: writer samples with softmax confidence < `WRITER_UNKNOWN_THRESHOLD` (default 0.9) are predicted as `-1`. This is the primary heuristic to tune.
- **`additional_train.csv`** contains samples with `writer_id=-1` (genuinely unknown writers) — not used in the baseline but relevant for training an unknown-writer detector.
- Checkpoints are Lightning `.ckpt` files (contain hparams + weights). The old `.pt` paths (`checkpoint.pt`, `checkpoint_best.pt`) from pre-Lightning runs are no longer produced.
- `log.json` is the contract between training and inference: it carries `label_map`, `idx_map`, `writer_unknown_threshold`, and `best_ckpt_path`.

## Baseline Results

Writer task (10 epochs, ResNet18, `lr=3e-4`, `batch_size=128`): **~92% validation accuracy** on a random 80/20 split of `train.csv`.
