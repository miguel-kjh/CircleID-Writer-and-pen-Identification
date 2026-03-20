import os


class Config:
    # Task
    TASK: str = "writer"

    # Model
    MODEL: str = "resnet18"

    # Paths
    DATASET: str = "raw"          # selects dataset/raw/ or dataset/raw_join/ etc.
    IMAGE_DIR: str = "dataset/"
    OUTPUT_DIR: str = "results/"

    @property
    def DATASET_DIR(self) -> str:
        return os.path.join("dataset", self.DATASET) + "/"

    # Hyperparameters
    EPOCHS: int = 10
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 3e-4
    IMG_SIZE: int = 224
    SEED: int = 0
    VAL_FRAC: float = 0.2

    # Writer task only: Below this confidence, writers are predicted as unknown (-1)
    WRITER_UNKNOWN_THRESHOLD: float = 0.9

    # Fine-tuning: path to a Lightning .ckpt whose backbone weights are pre-loaded
    PRETRAINED_CKPT: str | None = None

    @property
    def run_dir(self) -> str:
        lr_str = f"{self.LEARNING_RATE:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        ds_tag = f"_ds{self.DATASET.replace('/', '_')}" if self.DATASET != "raw" else ""
        ft_tag = "_ft" if self.PRETRAINED_CKPT else ""
        name = (
            f"{self.MODEL}_{self.TASK}{ds_tag}"
            f"_e{self.EPOCHS}_bs{self.BATCH_SIZE}_lr{lr_str}"
            f"_img{self.IMG_SIZE}_seed{self.SEED}{ft_tag}"
        )
        return os.path.join(self.OUTPUT_DIR, name)

    def setup(self):
        """Call after all attributes are set to create output dirs."""
        os.makedirs(self.run_dir, exist_ok=True)

    @property
    def ckpt_path(self) -> str:
        return os.path.join(self.run_dir, "checkpoint.pt")

    @property
    def best_ckpt_path(self) -> str:
        return os.path.join(self.run_dir, "checkpoint_best.pt")

    @property
    def log_path(self) -> str:
        return os.path.join(self.run_dir, "log.json")
