import pandas as pd


def generate_label_maps(train_df: pd.DataFrame, task: str):
    """Build label-index maps from the training data."""
    if task == "writer":
        labels = sorted(train_df["writer_id"].astype(str).unique().tolist())
    elif task == "pen":
        labels = sorted(train_df["pen_id"].astype(str).unique().tolist())
    else:
        raise ValueError("task must be 'writer' or 'pen'")

    label_map = {label: i for i, label in enumerate(labels)}
    index_map = {i: label for label, i in label_map.items()}

    return label_map, index_map


def random_split(df: pd.DataFrame, val_frac: float, seed: int):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    val_size = int(round(val_frac * len(df)))

    val_df = df.iloc[:val_size].copy()
    train_df = df.iloc[val_size:].copy()

    return train_df, val_df
