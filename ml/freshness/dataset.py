"""
Module: ml/freshness/dataset.py
Purpose: PyTorch Dataset and DataLoader loaders for Fresh vs. Stale produce images
"""

from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Supported produce classes in the Kaggle dataset
SUPPORTED_PRODUCE = ["apple", "banana", "bitter gourd", "capsicum", "orange", "tomato"]

def get_freshness_transforms(img_size: int = 224):
    """
    Returns train and validation/test transforms.
    Data augmentation is performed ONLY on training data to prevent leakage.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

class FreshnessDataset(Dataset):
    """
    Custom PyTorch Dataset for Fresh / Stale classification.
    Binary label: 0 = Fresh, 1 = Stale
    """
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), str(path)

def discover_freshness_samples(base_dir: Path, split: str = "train") -> List[Tuple[Path, int]]:
    """
    Discovers image files in the dataset folder.
    Supports either datasets/fresh_stale/processed/<split>/<class>
    or datasets/fresh_stale/<train_or_test>/<fresh_or_stale_categories>.
    """
    samples = []
    split_dir = base_dir / "processed" / split

    if split_dir.exists():
        fresh_dir = split_dir / "fresh"
        stale_dir = split_dir / "stale"
        if fresh_dir.exists():
            for p in fresh_dir.glob("*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    samples.append((p, 0)) # 0 = Fresh
        if stale_dir.exists():
            for p in stale_dir.glob("*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    samples.append((p, 1)) # 1 = Stale
    else:
        # Search direct subfolders
        all_imgs = [p for p in base_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        for p in all_imgs:
            p_str = str(p).lower()
            if split in p_str:
                if "fresh" in p_str and "stale" not in p_str:
                    samples.append((p, 0))
                elif "stale" in p_str or "rotten" in p_str:
                    samples.append((p, 1))

    return samples
