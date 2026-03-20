"""Main training script. Reproduces the baseline.ipynb training loop.

Usage:
    python train.py
    python train.py --task pen
    python train.py --task writer --epochs 1
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report as skl_report

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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
    parser.add_argument("--dataset",     default=cfg.DATASET, help="dataset subfolder under dataset/ (e.g. raw, raw_join)")
    parser.add_argument("--image-dir",   default=cfg.IMAGE_DIR)
    parser.add_argument("--output-dir",  default=cfg.OUTPUT_DIR)
    # Early stopping
    parser.add_argument("--early-stopping", action="store_true", default=False,
                        help="Enable early stopping")
    parser.add_argument("--es-monitor",  default="val/acc",
                        help="Metric to monitor for early stopping (default: val/acc)")
    parser.add_argument("--es-patience", type=int, default=5,
                        help="Epochs with no improvement before stopping (default: 5)")
    parser.add_argument("--es-min-delta", type=float, default=0.001,
                        help="Minimum improvement to reset patience (default: 0.001)")
    parser.add_argument("--es-mode", choices=["min", "max"], default="max",
                        help="Whether to minimize or maximize the monitored metric (default: max)")
    # Scheduler
    parser.add_argument("--scheduler", choices=["none", "cosine", "linear"], default="none",
                        help="LR scheduler: none | cosine (CosineAnnealingLR) | linear (LinearLR decay to 0)")
    parser.add_argument("--pretrained-ckpt", type=str, default=None,
                        help="Path to a Lightning .ckpt file whose backbone weights are loaded before fine-tuning (FC layer skipped due to class count mismatch).")
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
    cfg.EARLY_STOPPING           = args.early_stopping
    cfg.ES_MONITOR               = args.es_monitor
    cfg.ES_PATIENCE              = args.es_patience
    cfg.ES_MIN_DELTA             = args.es_min_delta
    cfg.ES_MODE                  = args.es_mode
    cfg.SCHEDULER                = args.scheduler
    cfg.PRETRAINED_CKPT          = args.pretrained_ckpt
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

    net = build_model(cfg.MODEL, num_classes=len(dm.label_map))

    if cfg.PRETRAINED_CKPT:
        ckpt = torch.load(cfg.PRETRAINED_CKPT, map_location="cpu")
        current_shapes = {k: v.shape for k, v in net.state_dict().items()}
        pretrained_state = {
            k[len("net."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("net.")
            and k[len("net."):] in current_shapes
            and v.shape == current_shapes[k[len("net."):]]
        }
        missing, unexpected = net.load_state_dict(pretrained_state, strict=False)
        print(f"Loaded pretrained backbone from: {cfg.PRETRAINED_CKPT}")
        print(f"  Missing keys (new FC expected): {missing}")
        print(f"  Unexpected keys: {unexpected}")

    module = CircleIDModule(net=net, lr=cfg.LEARNING_RATE, task=cfg.TASK,
                            idx_map=dm.idx_map,
                            writer_unknown_threshold=cfg.WRITER_UNKNOWN_THRESHOLD,
                            scheduler=cfg.SCHEDULER,
                            max_epochs=cfg.EPOCHS)

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

    callbacks = [best_ckpt_cb, last_ckpt_cb]
    if cfg.EARLY_STOPPING:
        callbacks.append(EarlyStopping(
            monitor=cfg.ES_MONITOR,
            patience=cfg.ES_PATIENCE,
            min_delta=cfg.ES_MIN_DELTA,
            mode=cfg.ES_MODE,
            verbose=True,
        ))

    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(module, datamodule=dm)

    # Update log.json with actual best checkpoint path (Lightning uses .ckpt extension)
    log = json.loads(Path(cfg.log_path).read_text())
    log["best_ckpt_path"] = best_ckpt_cb.best_model_path
    Path(cfg.log_path).write_text(json.dumps(log, indent=4))

    wandb_logger.experiment.summary["best_val_acc"] = best_ckpt_cb.best_model_score.item()

    print(f"Best checkpoint: {best_ckpt_cb.best_model_path}")

    # ── Val classification report ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_module = CircleIDModule.from_checkpoint(
        best_ckpt_cb.best_model_path,
        net_builder=lambda n: build_model(cfg.MODEL, n),
    )
    best_module.to(device).eval()

    idx_map = dm.idx_map  # int -> str
    has_unknown_class = "-1" in idx_map.values()
    threshold = cfg.WRITER_UNKNOWN_THRESHOLD
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y in dm.val_dataloader():
            x = x.to(device)
            logits = best_module(x)
            if cfg.TASK == "writer":
                probs = F.softmax(logits, dim=1)
                confs, indices = probs.max(dim=1)
                for conf, idx, true_idx in zip(confs.cpu(), indices.cpu(), y):
                    if has_unknown_class:
                        pred_label = idx_map[int(idx)]
                    else:
                        pred_label = "-1" if float(conf) < threshold else idx_map[int(idx)]
                    all_pred.append(pred_label)
                    all_true.append(idx_map[int(true_idx)])
            else:
                for idx, true_idx in zip(logits.argmax(1).cpu(), y):
                    all_pred.append(idx_map[int(idx)])
                    all_true.append(idx_map[int(true_idx)])

    report_str = skl_report(all_true, all_pred, zero_division=0)
    print("\nVal classification report:\n", report_str)

    report_txt = os.path.join(cfg.run_dir, "classification_report.txt")
    Path(report_txt).write_text(report_str, encoding="utf-8")

    report_dict = skl_report(all_true, all_pred, output_dict=True, zero_division=0)
    report_csv = os.path.join(cfg.run_dir, "classification_report.csv")
    pd.DataFrame(report_dict).T.reset_index().rename(
        columns={"index": "class"}
    ).to_csv(report_csv, index=False)

    print(f"Wrote val report → {report_txt}")
    print(f"Wrote val report → {report_csv}")
    # ───────────────────────────────────────────────────────────────────────────

    # ── Threshold optimization (known-only writer model) ──────────────────────
    if cfg.TASK == "writer" and not has_unknown_class:
        from torch.utils.data import DataLoader
        from src.data.dataset import CircleDataset

        # --- Confidences for KNOWN samples (val split of known_only) ---------
        known_confs = []
        with torch.no_grad():
            for x, _ in dm.val_dataloader():
                x = x.to(device)
                probs = F.softmax(best_module(x), dim=1)
                confs, _ = probs.max(dim=1)
                known_confs.extend(confs.cpu().tolist())

        # --- Confidences for UNKNOWN samples (multiclass val, writer_id==-1) -
        mc_val_csv = os.path.join("dataset", "process", "multiclass", "val.csv")
        mc_val_df  = pd.read_csv(mc_val_csv)
        unknown_df = mc_val_df[mc_val_df["writer_id"].astype(str) == "-1"].reset_index(drop=True)
        unknown_ds = CircleDataset(
            unknown_df, cfg.IMAGE_DIR, return_label=False, augment=False, img_size=cfg.IMG_SIZE
        )
        unknown_loader = DataLoader(
            unknown_ds, batch_size=cfg.BATCH_SIZE, num_workers=27, shuffle=False
        )
        unknown_confs = []
        with torch.no_grad():
            for x, _ in unknown_loader:
                x = x.to(device)
                probs = F.softmax(best_module(x), dim=1)
                confs, _ = probs.max(dim=1)
                unknown_confs.extend(confs.cpu().tolist())

        # --- Sweep 101 threshold values [0.0 … 1.0] --------------------------
        thresh_rows = []
        for t in np.linspace(0, 1, 101):
            tp = sum(1 for c in unknown_confs if c < t)     # unknown → -1  ✓
            fn = len(unknown_confs) - tp                    # unknown → known  ✗
            fp = sum(1 for c in known_confs if c < t)       # known → -1  ✗
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            thresh_rows.append({
                "threshold": round(float(t), 4),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 4),
                "recall":    round(recall,    4),
                "f1":        round(f1,        4),
            })

        thresh_df      = pd.DataFrame(thresh_rows)
        best_row       = thresh_df.loc[thresh_df["f1"].idxmax()]
        best_threshold = float(best_row["threshold"])

        print(f"\nThreshold sweep ({len(known_confs)} known, {len(unknown_confs)} unknown):")
        print(thresh_df[["threshold", "precision", "recall", "f1"]].to_string(index=False))
        print(f"\nBest threshold: {best_threshold:.2f}  (F1={best_row['f1']:.4f})")

        thresh_csv = os.path.join(cfg.run_dir, "threshold_optimization.csv")
        thresh_df.to_csv(thresh_csv, index=False)
        print(f"Wrote threshold sweep → {thresh_csv}")

        # Persist best threshold in log.json
        log = json.loads(Path(cfg.log_path).read_text())
        log["best_threshold"] = best_threshold
        Path(cfg.log_path).write_text(json.dumps(log, indent=4))
    # ───────────────────────────────────────────────────────────────────────────

    # Generate submission from best checkpoint
    dm.setup("predict")
    preds = trainer.predict(module, datamodule=dm,
                            ckpt_path=best_ckpt_cb.best_model_path)
    rows = [row for batch in preds for row in batch]

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
