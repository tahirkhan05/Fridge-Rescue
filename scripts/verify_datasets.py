"""
Script: scripts/verify_datasets.py
Purpose: Validate existence, integrity, splits, and class distribution of both mandatory datasets
Mandatory Datasets:
  1. Smart Refrigerator (Roboflow Universe)
  2. Fresh and Stale Images of Fruits and Vegetables (Kaggle)
"""

import sys
import yaml
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTION_DIR = PROJECT_ROOT / "datasets" / "smart_refrigerator"
FRESHNESS_DIR = PROJECT_ROOT / "datasets" / "fresh_stale"
SAMPLE_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "dataset_samples.png"

def verify_smart_refrigerator():
    print("\n" + "="*70)
    print("VERIFYING DATASET 1: SMART REFRIGERATOR (FOOD DETECTION)")
    print("="*70)

    if not DETECTION_DIR.exists():
        print("ERROR: Smart Refrigerator dataset not found at:", DETECTION_DIR)
        print("Download it using: python scripts/download_detection_dataset.py")
        return False, None

    # Check data.yaml
    yaml_candidates = list(DETECTION_DIR.rglob("data.yaml"))
    if not yaml_candidates:
        print("ERROR: data.yaml missing in", DETECTION_DIR)
        print("Download it using: python scripts/download_detection_dataset.py")
        return False, None

    data_yaml_path = yaml_candidates[0]
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    classes = meta.get("names", [])
    if isinstance(classes, dict):
        class_list = [classes[k] for k in sorted(classes.keys())]
    else:
        class_list = list(classes)

    splits = ["train", "valid", "test"]
    split_counts = {}
    corrupt_files = 0
    sample_images = []

    for s in splits:
        img_dir = DETECTION_DIR / s / "images"
        if not img_dir.exists():
            img_dir = DETECTION_DIR / s
        if not img_dir.exists():
            split_counts[s] = 0
            continue

        imgs = [f for f in img_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        valid_count = 0
        for p in imgs:
            try:
                with Image.open(p) as im:
                    im.verify()
                valid_count += 1
                if len(sample_images) < 4:
                    sample_images.append((p, f"Smart Fridge: {p.stem[:15]}"))
            except Exception:
                corrupt_files += 1
        split_counts[s] = valid_count

    total_images = sum(split_counts.values())
    if total_images == 0:
        print("ERROR: No valid images found in", DETECTION_DIR)
        print("Download it using: python scripts/download_detection_dataset.py")
        return False, None

    print(f"Dataset: Smart Refrigerator — Northumbria University Newcastle")
    print(f"Total images: {total_images}")
    print(f"Classes: {len(class_list)}")
    print(f"Sample classes (first 10 of {len(class_list)}): {class_list[:10]}")
    print(f"Missing/corrupt files: {corrupt_files}")
    print(f"Train: {split_counts.get('train', 0)}")
    print(f"Validation: {split_counts.get('valid', 0)}")
    print(f"Test: {split_counts.get('test', 0)}")
    print("STATUS: VERIFIED OK")

    return True, sample_images

def verify_fresh_stale():
    print("\n" + "="*70)
    print("VERIFYING DATASET 2: FRESH AND STALE (FRESHNESS CLASSIFICATION)")
    print("="*70)

    if not FRESHNESS_DIR.exists():
        print("ERROR: Fresh/Stale dataset not found at:", FRESHNESS_DIR)
        print("Download it using: python scripts/download_freshness_dataset.py")
        return False, None

    processed_dir = FRESHNESS_DIR / "processed"
    target_base = processed_dir if processed_dir.exists() else FRESHNESS_DIR

    splits = ["train", "val", "test"]
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_counts = {"fresh": 0, "stale": 0}
    corrupt_files = 0
    sample_images = []

    # If processed folder exists with train/val/test
    if processed_dir.exists():
        for s in splits:
            for c in ["fresh", "stale"]:
                folder = processed_dir / s / c
                if folder.exists():
                    files = [f for f in folder.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
                    for p in files:
                        try:
                            with Image.open(p) as im:
                                im.verify()
                            split_counts[s] += 1
                            class_counts[c] += 1
                            if len(sample_images) < 4:
                                sample_images.append((p, f"Fresh/Stale: {c} ({p.name[:12]})"))
                        except Exception:
                            corrupt_files += 1
    else:
        # Search all image files in FRESHNESS_DIR
        for p in FRESHNESS_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                label = "fresh" if "fresh" in str(p).lower() else ("stale" if "stale" in str(p).lower() or "rotten" in str(p).lower() else None)
                if label:
                    try:
                        with Image.open(p) as im:
                            im.verify()
                        class_counts[label] += 1
                        if len(sample_images) < 4:
                            sample_images.append((p, f"Fresh/Stale: {label}"))
                    except Exception:
                        corrupt_files += 1

    total_images = sum(class_counts.values())
    if total_images == 0:
        print("ERROR: Fresh/Stale dataset contains no valid images.")
        print("Download it using: python scripts/download_freshness_dataset.py")
        return False, None

    print(f"Dataset: Fresh and Stale Images of Fruits and Vegetables")
    print(f"Total images: {total_images}")
    print(f"Classes: 2 (fresh, stale) across 6 produce types (apple, banana, bitter gourd, capsicum, orange, tomato)")
    print(f"Images per class: Fresh={class_counts['fresh']}, Stale={class_counts['stale']}")
    print(f"Missing/corrupt files: {corrupt_files}")
    if processed_dir.exists():
        print(f"Train: {split_counts['train']}")
        print(f"Validation: {split_counts['val']}")
        print(f"Test: {split_counts['test']}")
    else:
        print(f"Splits: Not yet partitioned (run python scripts/download_freshness_dataset.py)")
    print("STATUS: VERIFIED OK")

    return True, sample_images

def generate_visualization(samples1, samples2):
    all_samples = (samples1 or [])[:4] + (samples2 or [])[:4]
    if not all_samples:
        return

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Fridge Rescue — Mandatory Dataset Verification Samples", fontsize=16, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx < len(all_samples):
            p, label = all_samples[idx]
            try:
                img = Image.open(p)
                ax.imshow(img)
                ax.set_title(label, fontsize=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center')
        ax.axis('off')

    plt.tight_layout()
    SAMPLE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAMPLE_OUTPUT_PATH, dpi=120)
    plt.close()
    print(f"\nSample visualization saved to: {SAMPLE_OUTPUT_PATH}")

def main():
    print("="*70)
    print("FRIDGE RESCUE — DATASET PRE-TRAINING INTEGRITY VALIDATION")
    print("="*70)

    ok1, samples1 = verify_smart_refrigerator()
    ok2, samples2 = verify_fresh_stale()

    if not ok1 or not ok2:
        print("\n" + "!"*70)
        print("VALIDATION FAILED: Both mandatory datasets must be present before training.")
        print("Run the corresponding download scripts and try again.")
        print("!"*70 + "\n")
        sys.exit(1)

    generate_visualization(samples1, samples2)

    print("\n" + "="*70)
    print("ALL MANDATORY DATASETS VALIDATED SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
