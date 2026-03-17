"""Inference script. Loads the best checkpoint and generates a submission CSV.

Usage:
    python predict.py
    python predict.py --task pen
"""

import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import CircleDataset
from src.models.resnet import build_model
from src.models.train import predict


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Predict CircleID baseline")
    parser.add_argument("--task",        choices=["writer", "pen"], default=cfg.TASK)
    parser.add_argument("--model",       default=cfg.MODEL)
    parser.add_argument("--batch-size",  type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--img-size",    type=int,   default=cfg.IMG_SIZE)
    parser.add_argument("--threshold",   type=float, default=cfg.WRITER_UNKNOWN_THRESHOLD)
    parser.add_argument("--dataset-dir", default=cfg.DATASET_DIR)
    parser.add_argument("--image-dir",   default=cfg.IMAGE_DIR)
    parser.add_argument("--output-dir",  default=cfg.OUTPUT_DIR)
    args = parser.parse_args()

    cfg.TASK                     = args.task
    cfg.MODEL                    = args.model
    cfg.BATCH_SIZE               = args.batch_size
    cfg.IMG_SIZE                 = args.img_size
    cfg.WRITER_UNKNOWN_THRESHOLD = args.threshold
    cfg.DATASET_DIR              = args.dataset_dir
    cfg.IMAGE_DIR                = args.image_dir
    cfg.OUTPUT_DIR               = args.output_dir
    cfg.setup()
    return cfg


def main():
    cfg = parse_args()

    if not os.path.exists(cfg.best_ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {cfg.best_ckpt_path}\n"
            f"Run `python train.py --task {cfg.TASK}` first."
        )

    with open(cfg.log_path, encoding="utf-8") as f:
        log = json.load(f)

    idx_map = {int(k): v for k, v in log["idx_map"].items()}
    label_map = log["label_map"]
    writer_unknown_threshold = log.get("writer_unknown_threshold", cfg.WRITER_UNKNOWN_THRESHOLD)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(label_map)).to(device)
    model_state = torch.load(cfg.best_ckpt_path, map_location=device)
    model.load_state_dict(model_state["model"])
    print(f"Loaded checkpoint: {cfg.best_ckpt_path}")

    test_df = pd.read_csv(os.path.join(cfg.DATASET_DIR, "test.csv"))

    test_ds = CircleDataset(test_df, img_root=cfg.IMAGE_DIR, return_label=False, augment=False, img_size=cfg.IMG_SIZE)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    predictions = predict(model, test_loader, device, idx_map, cfg.TASK, writer_unknown_threshold)

    if cfg.TASK == "writer":
        sub = pd.DataFrame(predictions, columns=["image_id", "writer_id"])
        out_name = os.path.join(cfg.OUTPUT_DIR, "submission_writer.csv")
    else:
        sub = pd.DataFrame(predictions, columns=["image_id", "pen_id"])
        out_name = os.path.join(cfg.OUTPUT_DIR, "submission_pen.csv")

    sub.to_csv(out_name, index=False)
    print(f"Wrote: {out_name}")


if __name__ == "__main__":
    main()
