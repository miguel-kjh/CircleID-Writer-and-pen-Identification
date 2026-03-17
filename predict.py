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

import src.config as cfg
from src.data.dataset import CircleDataset
from src.models.resnet import build_model
from src.models.train import predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["writer", "pen"], default=None,
                        help="Override config.TASK")
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task if args.task is not None else cfg.TASK

    best_ckpt_path = f"{cfg.OUTPUT_DIR}/baseline_{task}_best.pt"
    log_path       = f"{cfg.OUTPUT_DIR}/log_{task}.json"

    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {best_ckpt_path}\n"
            f"Run `python train.py --task {task}` first."
        )

    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)

    idx_map = {int(k): v for k, v in log["idx_map"].items()}
    label_map = log["label_map"]
    writer_unknown_threshold = log.get("writer_unknown_threshold", cfg.WRITER_UNKNOWN_THRESHOLD)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(label_map)).to(device)
    model_state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(model_state["model"])
    print(f"Loaded checkpoint: {best_ckpt_path}")

    test_df = pd.read_csv(os.path.join(cfg.DATASET_DIR, "test.csv"))

    test_ds = CircleDataset(test_df, img_root=cfg.IMAGE_DIR, return_label=False, augment=False, img_size=cfg.IMG_SIZE)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    predictions = predict(model, test_loader, device, idx_map, task, writer_unknown_threshold)

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
