import os


class Config:
    # Task
    TASK: str = "writer"

    # Paths
    DATASET_DIR: str = "dataset/raw/"
    IMAGE_DIR: str = "dataset/"
    OUTPUT_DIR: str = "results/"

    # Hyperparameters
    EPOCHS: int = 10
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 3e-4
    IMG_SIZE: int = 224
    SEED: int = 0
    VAL_FRAC: float = 0.2

    # Writer task only: Below this confidence, writers are predicted as unknown (-1)
    WRITER_UNKNOWN_THRESHOLD: float = 0.9

    def setup(self):
        """Call after all attributes are set to create output dirs."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

    @property
    def ckpt_path(self) -> str:
        return f"{self.OUTPUT_DIR}/baseline_{self.TASK}.pt"

    @property
    def best_ckpt_path(self) -> str:
        return f"{self.OUTPUT_DIR}/baseline_{self.TASK}_best.pt"

    @property
    def log_path(self) -> str:
        return f"{self.OUTPUT_DIR}/log_{self.TASK}.json"
