# CircleID Writer and Pen Identification

PyTorch pipeline for the [ICDAR 2026 CircleID Writer Identification](https://www.kaggle.com/competitions/icdar-2026-circleid-writer-identification/data) Kaggle competition. The goal is to classify the **writer** (W01–W51, or `-1` for unknown) and/or **pen** (1–8) from handwritten circle samples.

## Setup

```bash
pip install torch torchvision numpy pandas pillow tqdm wandb
```

## Usage

```bash
# Train
python train.py --task writer --model resnet18 --epochs 10 --batch-size 128 --lr 3e-4
python train.py --task pen

# Inference (requires a prior training run)
python predict.py --task writer --model resnet18
python predict.py --task pen
```

Checkpoints, logs, and submission CSVs are saved under `results/<run_name>/` where the run name encodes the hyperparameters (e.g. `resnet18_writer_e10_bs128_lr3e-4_img224_seed0`).

## Data

Place the competition files under `dataset/`:

```
dataset/
  raw/        # train.csv, test.csv, additional_train.csv, sample_submission.csv
  images/     # PNG handwriting samples
```

## Architecture

```
src/
  config.py          # Config class with hyperparameters and path properties
  utils.py           # set_seeds
  data/
    dataset.py       # CircleDataset (PyTorch Dataset)
    utils.py         # generate_label_maps, random_split
  models/
    __init__.py      # Model registry + build_model factory
    base.py          # BaseModel abstract class
    resnet.py        # ResNet18 implementation
    train.py         # train_epoch, evaluate, predict functions
```

New models can be added by subclassing `BaseModel`, setting a `NAME` attribute, and registering in `src/models/__init__.py`.

## Baseline Results

| Task   | Model    | Epochs | Val Accuracy |
|--------|----------|--------|--------------|
| Writer | ResNet18 | 10     | ~92%         |

Experiment tracking via [Weights & Biases](https://wandb.ai) (project: `circleid`).

## References

- [Competition page](https://www.kaggle.com/competitions/icdar-2026-circleid-writer-identification/data)
