#!/usr/bin/env python3
"""Prepare stratified train/val splits for multiclass and binary tasks."""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Prepare stratified dataset splits.")
    p.add_argument("--source", default="dataset/raw_join/train.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    return p.parse_args()


def print_distribution(name, df):
    counts = df["writer_id"].value_counts().sort_index()
    print(f"\n{name} ({len(df)} rows) — writer_id distribution:")
    for wid, cnt in counts.items():
        print(f"  {wid}: {cnt}")


def main():
    args = parse_args()

    df = pd.read_csv(args.source)
    print(f"Loaded {len(df)} rows from {args.source}")

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_frac,
        random_state=args.seed,
        stratify=df["writer_id"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Multiclass splits (writer_id unchanged)
    mc_dir = "dataset/process/multiclass"
    os.makedirs(mc_dir, exist_ok=True)
    train_df.to_csv(os.path.join(mc_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(mc_dir, "val.csv"), index=False)
    print(f"\nWrote multiclass splits to {mc_dir}/")

    # Binary splits (known writer → 1, unknown → -1)
    bin_train = train_df.copy()
    bin_val = val_df.copy()
    bin_train["writer_id"] = bin_train["writer_id"].apply(lambda x: -1 if str(x) == "-1" else 1)
    bin_val["writer_id"] = bin_val["writer_id"].apply(lambda x: -1 if str(x) == "-1" else 1)

    bin_dir = "dataset/process/binary"
    os.makedirs(bin_dir, exist_ok=True)
    bin_train.to_csv(os.path.join(bin_dir, "train.csv"), index=False)
    bin_val.to_csv(os.path.join(bin_dir, "val.csv"), index=False)
    print(f"Wrote binary splits to {bin_dir}/")

    # Known-only splits (filter out unknown writers, keep only known 44 classes)
    known_train = train_df[train_df["writer_id"].astype(str) != "-1"].reset_index(drop=True)
    known_val   = val_df[val_df["writer_id"].astype(str) != "-1"].reset_index(drop=True)

    ko_dir = "dataset/process/known_only"
    os.makedirs(ko_dir, exist_ok=True)
    known_train.to_csv(os.path.join(ko_dir, "train.csv"), index=False)
    known_val.to_csv(os.path.join(ko_dir, "val.csv"), index=False)
    print(f"Wrote known_only splits to {ko_dir}/")

    # Sanity check distributions
    print_distribution("multiclass/train", train_df)
    print_distribution("multiclass/val", val_df)
    print_distribution("binary/train", bin_train)
    print_distribution("binary/val", bin_val)
    print_distribution("known_only/train", known_train)
    print_distribution("known_only/val", known_val)


if __name__ == "__main__":
    main()
