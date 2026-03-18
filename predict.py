"""Inference script. Loads the best checkpoint and generates a submission CSV.

Usage:
    python predict.py
    python predict.py --task pen
"""

import argparse
import json
import os

import pandas as pd
import pytorch_lightning as pl

from src.config import Config
from src.data.datamodule import CircleDataModule
from src.models import _REGISTRY, build_model
from src.models.lightning_module import CircleIDModule


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Predict CircleID baseline")
    parser.add_argument("--task",        choices=["writer", "pen"], default=cfg.TASK)
    parser.add_argument("--model",       choices=list(_REGISTRY), default=cfg.MODEL)
    parser.add_argument("--epochs",      type=int,   default=cfg.EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--img-size",    type=int,   default=cfg.IMG_SIZE)
    parser.add_argument("--seed",        type=int,   default=cfg.SEED)
    parser.add_argument("--val-frac",    type=float, default=cfg.VAL_FRAC)
    parser.add_argument("--threshold",   type=float, default=cfg.WRITER_UNKNOWN_THRESHOLD)
    parser.add_argument("--dataset",     default=cfg.DATASET, help="dataset subfolder under dataset/ (e.g. raw, raw_join)")
    parser.add_argument("--image-dir",   default=cfg.IMAGE_DIR)
    parser.add_argument("--output-dir",  default=cfg.OUTPUT_DIR)
    args = parser.parse_args()

    cfg.TASK                     = args.task
    cfg.MODEL                    = args.model
    cfg.EPOCHS                   = args.epochs
    cfg.BATCH_SIZE               = args.batch_size
    cfg.LEARNING_RATE            = args.lr
    cfg.IMG_SIZE                 = args.img_size
    cfg.SEED                     = args.seed
    cfg.VAL_FRAC                 = args.val_frac
    cfg.WRITER_UNKNOWN_THRESHOLD = args.threshold
    cfg.DATASET                  = args.dataset
    cfg.IMAGE_DIR                = args.image_dir
    cfg.OUTPUT_DIR               = args.output_dir
    cfg.setup()
    return cfg


def main():
    cfg = parse_args()

    with open(cfg.log_path, encoding="utf-8") as f:
        log = json.load(f)

    ckpt_path = log.get("best_ckpt_path", cfg.best_ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run `python train.py --task {cfg.TASK}` first."
        )

    module = CircleIDModule.from_checkpoint(
        ckpt_path,
        net_builder=lambda n: build_model(cfg.MODEL, n),
    )

    dm = CircleDataModule(cfg)
    dm.setup("predict")

    trainer = pl.Trainer(enable_progress_bar=True, logger=False)
    preds = trainer.predict(module, datamodule=dm, ckpt_path=ckpt_path)
    rows = [row for batch in preds for row in batch]

    if cfg.TASK == "writer":
        sub = pd.DataFrame(rows, columns=["image_id", "writer_id"])
        out_name = os.path.join(cfg.OUTPUT_DIR, f"submission_writer_{cfg.MODEL}.csv")
    else:
        sub = pd.DataFrame(rows, columns=["image_id", "pen_id"])
        out_name = os.path.join(cfg.OUTPUT_DIR, f"submission_pen_{cfg.MODEL}.csv")

    sub.to_csv(out_name, index=False)
    print(f"Wrote: {out_name}")


if __name__ == "__main__":
    main()
