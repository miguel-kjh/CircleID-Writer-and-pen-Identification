import json
import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import CircleDataset
from src.data.utils import generate_label_maps, random_split


class CircleDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.label_map = None   # str -> int, populated in setup("fit")
        self.idx_map   = None   # int -> str
        self._train_ds = self._val_ds = self._test_ds = None

    def setup(self, stage: str):
        cfg = self.cfg
        if stage == "fit":
            train_df = pd.read_csv(os.path.join(cfg.DATASET_DIR, "train.csv"))
            self.label_map, self.idx_map = generate_label_maps(train_df, cfg.TASK)
            col = "writer_id" if cfg.TASK == "writer" else "pen_id"
            train_df["y"] = train_df[col].astype(str).map(self.label_map).astype(int)
            train_df, val_df = random_split(train_df, cfg.VAL_FRAC, cfg.SEED)
            self._train_ds = CircleDataset(train_df, cfg.IMAGE_DIR, return_label=True,  augment=True,  img_size=cfg.IMG_SIZE)
            self._val_ds   = CircleDataset(val_df,   cfg.IMAGE_DIR, return_label=True,  augment=False, img_size=cfg.IMG_SIZE)
            # Write log.json (backward compat + human readable)
            log = {
                "task": cfg.TASK,
                "seed": cfg.SEED,
                "label_map": self.label_map,
                "idx_map": self.idx_map,
                "writer_unknown_threshold": cfg.WRITER_UNKNOWN_THRESHOLD,
                "val_frac": cfg.VAL_FRAC,
            }
            Path(cfg.log_path).write_text(json.dumps(log, indent=4), encoding="utf-8")

        elif stage == "predict":
            test_df = pd.read_csv(os.path.join(cfg.DATASET_DIR, "test.csv"))
            self._test_ds = CircleDataset(test_df, cfg.IMAGE_DIR, return_label=False, augment=False, img_size=cfg.IMG_SIZE)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.cfg.BATCH_SIZE, shuffle=shuffle,
                          pin_memory=torch.cuda.is_available(), drop_last=False)

    def train_dataloader(self):
        return self._loader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self._val_ds, shuffle=False)

    def predict_dataloader(self):
        return self._loader(self._test_ds, shuffle=False)
