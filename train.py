"""Main training script. Reproduces the baseline.ipynb training loop.

Usage:
    python train.py
    python train.py --task pen
    python train.py --task writer --epochs 1
"""

import json
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.config import Config
from src.data.datamodule import CircleDataModule
from src.models import _REGISTRY, build_model
from src.models.lightning_module import CircleIDModule
from src.utils import set_seeds


def parse_args() -> Config:
    import argparse
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train CircleID baseline")
    parser.add_argument("--task",        choices=["writer", "pen"], default=cfg.TASK)
    parser.add_argument("--model",       choices=list(_REGISTRY), default=cfg.MODEL)
    parser.add_argument("--epochs",      type=int,   default=cfg.EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--img-size",    type=int,   default=cfg.IMG_SIZE)
    parser.add_argument("--seed",        type=int,   default=cfg.SEED)
    parser.add_argument("--val-frac",    type=float, default=cfg.VAL_FRAC)
    parser.add_argument("--threshold",   type=float, default=cfg.WRITER_UNKNOWN_THRESHOLD)
    parser.add_argument("--dataset-dir", default=cfg.DATASET_DIR)
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
    cfg.DATASET_DIR              = args.dataset_dir
    cfg.IMAGE_DIR                = args.image_dir
    cfg.OUTPUT_DIR               = args.output_dir
    cfg.setup()
    return cfg


def main():
    cfg = parse_args()
    print(f"Run dir: {cfg.run_dir}")

    set_seeds(cfg.SEED)

    dm = CircleDataModule(cfg)
    dm.setup("fit")             # populate label_map/idx_map before building module
    print(f"Train samples: {len(dm._train_ds)} | Validation samples: {len(dm._val_ds)} ({cfg.VAL_FRAC:.2f})")
    if cfg.TASK == "writer":
        print("Note: Validation accuracy is calculated only on known writers.")

    net    = build_model(cfg.MODEL, num_classes=len(dm.label_map))
    module = CircleIDModule(net=net, lr=cfg.LEARNING_RATE, task=cfg.TASK,
                            idx_map=dm.idx_map,
                            writer_unknown_threshold=cfg.WRITER_UNKNOWN_THRESHOLD)

    wandb_logger = WandbLogger(
        project="circleid",
        name=os.path.basename(cfg.run_dir),
        config={
            "task": cfg.TASK,
            "model": cfg.MODEL,
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "learning_rate": cfg.LEARNING_RATE,
            "img_size": cfg.IMG_SIZE,
            "seed": cfg.SEED,
            "val_frac": cfg.VAL_FRAC,
            "writer_unknown_threshold": cfg.WRITER_UNKNOWN_THRESHOLD,
            "num_classes": len(dm.label_map),
        },
    )

    best_ckpt_cb = ModelCheckpoint(
        dirpath=cfg.run_dir,
        filename="checkpoint_best",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
    )
    last_ckpt_cb = ModelCheckpoint(
        dirpath=cfg.run_dir,
        filename="checkpoint",
        every_n_epochs=1,
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        logger=wandb_logger,
        callbacks=[best_ckpt_cb, last_ckpt_cb],
        deterministic=True,
    )

    trainer.fit(module, datamodule=dm)

    # Update log.json with actual best checkpoint path (Lightning uses .ckpt extension)
    log = json.loads(Path(cfg.log_path).read_text())
    log["best_ckpt_path"] = best_ckpt_cb.best_model_path
    Path(cfg.log_path).write_text(json.dumps(log, indent=4))

    wandb_logger.experiment.summary["best_val_acc"] = best_ckpt_cb.best_model_score.item()

    print(f"Best checkpoint: {best_ckpt_cb.best_model_path}")

    # Generate submission from best checkpoint
    dm.setup("predict")
    preds = trainer.predict(module, datamodule=dm,
                            ckpt_path=best_ckpt_cb.best_model_path)
    rows = [row for batch in preds for row in batch]

    import pandas as pd
    name_model = os.path.basename(cfg.run_dir)
    if cfg.TASK == "writer":
        sub = pd.DataFrame(rows, columns=["image_id", "writer_id"])
        out_name = os.path.join(cfg.run_dir, f"submission_writer_{name_model}.csv")
    else:
        sub = pd.DataFrame(rows, columns=["image_id", "pen_id"])
        out_name = os.path.join(cfg.run_dir, f"submission_pen_{name_model}.csv")

    sub.to_csv(out_name, index=False)
    print(f"Wrote: {out_name}")


if __name__ == "__main__":
    main()
