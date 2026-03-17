from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights


class CircleDataset(Dataset):
    """
    Dataset for circles.
    - df: Pandas DataFrame
    - img_root: Root directory for image paths
    - return_label: Return labels as (x, y); otherwise return (x, image_id)
    - augment: Apply augmentation (should only be enabled for training)
    - img_size: Resize target (default 224)
    """

    def __init__(self, df: pd.DataFrame, img_root: Path, return_label: bool, augment: bool, img_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.return_label = return_label

        mean = ResNet18_Weights.DEFAULT.transforms().mean
        std = ResNet18_Weights.DEFAULT.transforms().std

        if augment:
            self.transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        image_id = str(row["image_id"])
        rel_path = str(row["image_path"])
        img_path = self.img_root / rel_path

        img = Image.open(img_path).convert("RGB")
        x = self.transforms(img)

        if self.return_label:
            y = int(row["y"])
            return x, y
        else:
            return x, image_id
