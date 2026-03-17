"""Main training script. Reproduces the baseline.ipynb training loop.

Usage:
    python train.py
    python train.py --task pen
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.config as cfg
from src.data.dataset import CircleDataset
from src.data.utils import generate_label_maps, random_split
from src.models.resnet import build_model
from src.models.train import evaluate, predict, train_epoch
from src.utils import set_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["writer", "pen"], default=None,
                        help="Override config.TASK")
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow CLI override of task
    task = args.task if args.task is not None else cfg.TASK

    # Re-derive output paths if task was overridden
    ckpt_path      = f"{cfg.OUTPUT_DIR}/baseline_{task}.pt"
    best_ckpt_path = f"{cfg.OUTPUT_DIR}/baseline_{task}_best.pt"
    log_path       = f"{cfg.OUTPUT_DIR}/log_{task}.json"

    set_seeds(cfg.SEED)

    train_df = pd.read_csv(os.path.join(cfg.DATASET_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(cfg.DATASET_DIR, "test.csv"))

    label_map, idx_map = generate_label_maps(train_df, task)

    if task == "writer":
        train_df["y"] = train_df["writer_id"].astype(str).map(label_map).astype(int)
    else:
        train_df["y"] = train_df["pen_id"].astype(str).map(label_map).astype(int)

    train_df, val_df = random_split(train_df, val_frac=cfg.VAL_FRAC, seed=cfg.SEED)
    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)} ({cfg.VAL_FRAC:.2f})")
    if task == "writer":
        print("Note: Validation accuracy is calculated only on known writers.")

    log = {
        "task": task,
        "seed": cfg.SEED,
        "label_map": label_map,
        "idx_map": idx_map,
        "writer_unknown_threshold": cfg.WRITER_UNKNOWN_THRESHOLD,
        "val_frac": cfg.VAL_FRAC,
    }
    Path(log_path).write_text(json.dumps(log, indent=4), encoding="utf-8")
    print(f"Saved log to {log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(label_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    train_ds = CircleDataset(train_df, img_root=cfg.IMAGE_DIR, return_label=True, augment=True, img_size=cfg.IMG_SIZE)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_ds = CircleDataset(val_df, img_root=cfg.IMAGE_DIR, return_label=True, augment=False, img_size=cfg.IMG_SIZE)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    best_acc = -1.0
    for epoch in range(cfg.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch + 1}/{cfg.EPOCHS}] Train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict()}, best_ckpt_path)

    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"Saved last checkpoint: {ckpt_path}")
    print(f"Saved best checkpoint: {best_ckpt_path} (best val acc={best_acc:.4f})")

    # Generate submission from best checkpoint
    model_state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(model_state["model"])

    test_ds = CircleDataset(test_df, img_root=cfg.IMAGE_DIR, return_label=False, augment=False, img_size=cfg.IMG_SIZE)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    predictions = predict(model, test_loader, device, idx_map, task, cfg.WRITER_UNKNOWN_THRESHOLD)

    if task == "writer":
        sub = pd.DataFrame(predictions, columns=["image_id", "writer_id"])
        out_name = os.path.join(cfg.OUTPUT_DIR, "submission_writer.csv")
    else:
        sub = pd.DataFrame(predictions, columns=["image_id", "pen_id"])
        out_name = os.path.join(cfg.OUTPUT_DIR, "submission_pen.csv")

    sub.to_csv(out_name, index=False)
    print(f"Wrote: {out_name}")


if __name__ == "__main__":
    main()
