# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition: [ICDAR 2026 CircleID Writer Identification](https://www.kaggle.com/competitions/icdar-2026-circleid-writer-identification/data). The goal is to identify the **writer** and/or **pen** used in handwriting samples (circle drawings).

Two tasks:
- **Writer Identification**: Classify which writer (W01–W51) wrote the sample, with an "unknown writer" option (`-1`) via confidence threshold.
- **Pen Classification**: Classify which pen (1–8) was used.

## Running the Notebook

```bash
pip install numpy pandas pillow torch torchvision tqdm
jupyter notebook baseline.ipynb
```

The notebook is designed to run both locally and on Kaggle. Switch paths via the `DATASET_DIR` / `OUTPUT_DIR` config variables at the top.

## Data Layout

```
icdar-2026-circleid-writer-identification/
  train.csv              # image_id, image_path, writer_id, pen_id
  additional_train.csv   # same schema; writer_id=-1 means unknown writer
  test.csv               # image_id, image_path (no labels)
  sample_submission.csv  # image_id, writer_id (example format)
  images/                # PNG handwriting samples (00001.png, etc.)
```

Train images use numeric filenames (`00001.png`). Test images use hex-string filenames (`v2_<hash>.png`). `image_path` in each CSV is relative to the dataset root.

## Notebook Architecture (`baseline.ipynb`)

The notebook is a single-file pipeline:

1. **Config block** – `TASK`, `DATASET_DIR`, `OUTPUT_DIR`, hyperparameters (`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `IMG_SIZE`, `SEED`, `VAL_FRAC`, `WRITER_UNKNOWN_THRESHOLD`).
2. **Utilities** – `set_seeds`, `generate_label_maps`, `random_split`.
3. **`CircleDataset`** – `torch.utils.data.Dataset` that reads images from disk via paths in the CSV, applies ImageNet normalization (from `ResNet18_Weights.DEFAULT`), and optionally applies augmentation (random rotation ±10°).
4. **Model / training** – `build_model` replaces ResNet18's `fc` head with a linear layer sized to the number of classes. `train_epoch` / `evaluate` / `predict` are standard PyTorch loops.
5. **Training loop** – Saves best checkpoint by validation accuracy (`baseline_{task}_best.pt`) and last checkpoint (`baseline_{task}.pt`). Also writes a `log_{task}.json` with label maps.
6. **Submission** – `predict()` applies a softmax confidence threshold for unknown-writer detection (writer task only). Outputs `submission_writer.csv` or `submission_pen.csv`.

## Key Design Decisions

- **Unknown writer**: Samples with softmax confidence < `WRITER_UNKNOWN_THRESHOLD` are predicted as `-1`. This threshold is the primary baseline heuristic to improve.
- **`additional_train.csv`** contains images where `writer_id=-1` (genuinely unknown writers) — useful for training the unknown-writer detector but not used in the current baseline.
- Label maps are built from `train.csv` only; `idx_map` (int → label string) is used at prediction time.
- The model always predicts from `BEST_CKPT_PATH` (best val acc), not the last checkpoint.
