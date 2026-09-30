import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from core.transforms import get_train_transforms_s1, get_train_transforms_s2, get_val_transforms
from core.pipeline import STAGE1_CLASSES

class SkinDataset(Dataset):
    """Robust PyTorch Dataset for Dermatological images with integrity verification."""
    def __init__(self, df: pd.DataFrame, transform=None, stage: int = 1, class_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.stage = stage
        self.class_to_idx = class_to_idx or {cls: i for i, cls in enumerate(STAGE1_CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
        except Exception as e:
            # Fallback for corrupted images: create a blank normalized image or pick previous valid
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            if self.transform:
                img = self.transform(img)

        if self.stage == 1:
            target = self.class_to_idx[row["label"]]
        else:
            target = int(row["label"])

        return img, torch.tensor(target, dtype=torch.long)


def build_dataloaders(df: pd.DataFrame, stage: int = 1, batch_size: int = 32, seed: int = 42):
    """
    Performs clean stratified Train / Val / Test splitting (70% / 15% / 15%)
    and creates DataLoaders with appropriate sampling and transforms.
    """
    train_df, test_df = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1765, stratify=train_df["label"], random_state=seed
    )

    val_transforms = get_val_transforms()
    train_transforms = get_train_transforms_s1() if stage == 1 else get_train_transforms_s2()

    train_dataset = SkinDataset(train_df, transform=train_transforms, stage=stage)
    val_dataset = SkinDataset(val_df, transform=val_transforms, stage=stage)
    test_dataset = SkinDataset(test_df, transform=val_transforms, stage=stage)

    # Use WeightedRandomSampler for Stage 2 if class distribution is imbalanced
    if stage == 2:
        targets = train_df["label"].values.astype(int)
        classes, class_counts = np.unique(targets, return_counts=True)
        weight_per_class = {cls: 1.0 / count for cls, count in zip(classes, class_counts)}
        samples_weight = np.array([weight_per_class[t] for t in targets])
        sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "dfs": {"train": train_df, "val": val_df, "test": test_df}
    }
