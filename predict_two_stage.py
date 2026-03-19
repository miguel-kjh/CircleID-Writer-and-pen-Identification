"""Two-stage inference for writer identification.

Stage 1: reads binary classifier results from an existing submission CSV
          (known=1 vs unknown=-1, produced by predict.py on process/binary dataset)
Stage 2: known-only classifier (W01–W51) applied to samples predicted as known

Usage:
    python predict_two_stage.py
    python predict_two_stage.py --binary-run-dir results/... --known-run-dir results/...
    python predict_two_stage.py --binary-csv results/.../submission_writer_....csv
"""

import argparse
import json
import os

import pandas as pd
import pytorch_lightning as pl

from src.config import Config
from src.data.datamodule import CircleDataModule
from src.models import build_model
from src.models.lightning_module import CircleIDModule

DEFAULT_BINARY_RUN_DIR = (
    "results/resnet18_writer_dsprocess_binary_e10_bs128_lr3e-4_img224_seed0"
)
DEFAULT_KNOWN_RUN_DIR = (
    "results/resnet18_writer_dsprocess_known_only_e10_bs128_lr3e-4_img224_seed0"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage writer identification inference")
    parser.add_argument("--binary-run-dir", default=DEFAULT_BINARY_RUN_DIR,
                        help="Run dir of the binary classifier (used to locate its submission CSV)")
    parser.add_argument("--binary-csv",     default=None,
                        help="Path to binary submission CSV directly (overrides --binary-run-dir)")
    parser.add_argument("--known-run-dir",  default=DEFAULT_KNOWN_RUN_DIR)
    parser.add_argument("--batch-size",     type=int, default=128)
    parser.add_argument("--img-size",       type=int, default=224)
    return parser.parse_args()


def find_binary_csv(run_dir: str) -> str:
    basename = os.path.basename(run_dir)
    candidate = os.path.join(run_dir, f"submission_writer_{basename}.csv")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Binary submission CSV not found: {candidate}\n"
            f"Run `python predict.py --dataset process/binary` first, or pass --binary-csv."
        )
    return candidate


def load_ckpt_path(run_dir: str) -> str:
    log_path = os.path.join(run_dir, "log.json")
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    ckpt_path = log.get("best_ckpt_path", os.path.join(run_dir, "checkpoint_best.pt"))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def model_name_from_run_dir(run_dir: str) -> str:
    return os.path.basename(run_dir).split("_")[0]


def main():
    args = parse_args()

    # --- Stage 1: load binary results from CSV ---
    binary_csv = args.binary_csv or find_binary_csv(args.binary_run_dir)
    binary_df = pd.read_csv(binary_csv, dtype=str)
    print(f"Loaded binary predictions from: {binary_csv} ({len(binary_df)} rows)")

    # --- Stage 2: run known-only classifier ---
    known_ckpt       = load_ckpt_path(args.known_run_dir)
    known_model_name = model_name_from_run_dir(args.known_run_dir)
    known_module     = CircleIDModule.from_checkpoint(
        known_ckpt,
        net_builder=lambda n: build_model(known_model_name, n),
    )
    # Disable threshold: known/unknown decision is delegated entirely to the binary
    # classifier. The known-only model should always return the argmax class.
    known_module.hparams.writer_unknown_threshold = 0.0

    cfg = Config()
    cfg.DATASET    = "raw"
    cfg.BATCH_SIZE = args.batch_size
    cfg.IMG_SIZE   = args.img_size

    dm = CircleDataModule(cfg)
    dm.setup("predict")

    trainer = pl.Trainer(enable_progress_bar=True, logger=False)

    print("Running known-only classifier...")
    preds = trainer.predict(known_module, datamodule=dm, ckpt_path=known_ckpt)
    known_rows = [row for batch in preds for row in batch]
    known_dict = {image_id: label for image_id, label in known_rows}

    # --- Combine ---
    combined = []
    for _, row in binary_df.iterrows():
        image_id    = row["image_id"]
        binary_pred = row["writer_id"]
        if binary_pred == "-1":
            final = "-1"
        else:
            final = known_dict[image_id]
        combined.append((image_id, final))

    n_unknown = sum(1 for _, label in combined if label == "-1")
    n_known   = len(combined) - n_unknown
    print(f"Routing: {n_known} known, {n_unknown} unknown out of {len(combined)} total")

    binary_basename = os.path.basename(args.binary_run_dir)
    known_basename  = os.path.basename(args.known_run_dir)
    out_name = os.path.join(
        args.binary_run_dir,
        f"submission_two_stage_{binary_basename}_x_{known_basename}.csv",
    )

    sub = pd.DataFrame(combined, columns=["image_id", "writer_id"])
    sub.to_csv(out_name, index=False)
    print(f"Wrote: {out_name}")


if __name__ == "__main__":
    main()
