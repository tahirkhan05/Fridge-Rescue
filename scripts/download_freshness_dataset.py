"""
Script: scripts/download_freshness_dataset.py
Purpose: Download and extract the mandatory Fresh and Stale fruits/vegetables dataset from Kaggle
Dataset: Fresh and Stale Images of Fruits and Vegetables
Author: Raghav R Potdar
Source: https://www.kaggle.com/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables
License: CC0 Public Domain
Target location: datasets/fresh_stale/
"""

import os
import sys
import zipfile
import shutil
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "fresh_stale"
DATASET_SLUG = "raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables"

SUPPORTED_FOOD_TYPES = [
    "apple",
    "banana",
    "bitter gourd",
    "capsicum",
    "orange",
    "tomato"
]

def setup_kaggle_credentials():
    token = os.getenv("KAGGLE_API_TOKEN")
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
        # Also ensure ~/.kaggle/access_token exists for CLI fallback
        home_kaggle = Path.home() / ".kaggle"
        home_kaggle.mkdir(parents=True, exist_ok=True)
        token_file = home_kaggle / "access_token"
        if not token_file.exists():
            try:
                token_file.write_text(token.strip(), encoding="utf-8")
            except Exception as e:
                print(f"Notice: Could not write token file: {e}")
        return True

    # Check if ~/.kaggle/kaggle.json or access_token already exists
    home_kaggle = Path.home() / ".kaggle"
    if (home_kaggle / "kaggle.json").exists() or (home_kaggle / "access_token").exists():
        return True

    print("\n" + "="*70)
    print("ERROR: Kaggle credentials not found.")
    print("Please set KAGGLE_API_TOKEN in your .env file or environment:")
    print("  KAGGLE_API_TOKEN=your_token_here")
    print("Or place your kaggle.json in ~/.kaggle/kaggle.json")
    print("="*70 + "\n")
    return False

def download_and_extract_freshness_dataset():
    print("="*70)
    print("FRIDGE RESCUE — DATASET 2: FRESH & STALE DATASET DOWNLOAD")
    print(f"Source: https://www.kaggle.com/{DATASET_SLUG}")
    print("License: CC0 Public Domain")
    print(f"Destination: {DATASET_DIR}")
    print("="*70)

    if not setup_kaggle_credentials():
        sys.exit(1)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download using Kaggle API
    print("\n[1/5] Authenticating with Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("  Kaggle authentication successful.")
    except Exception as e:
        print(f"ERROR authenticating with Kaggle API: {e}")
        sys.exit(1)

    print(f"\n[2/5] Downloading dataset '{DATASET_SLUG}'...")
    try:
        api.dataset_download_files(DATASET_SLUG, path=str(DATASET_DIR), quiet=False, unzip=False)
    except Exception as e:
        print(f"ERROR during download: {e}")
        sys.exit(1)

    # Step 2: Locate downloaded archive
    print("\n[3/5] Locating downloaded archive...")
    zip_files = list(DATASET_DIR.glob("*.zip"))
    if not zip_files:
        print(f"ERROR: No .zip archive found in {DATASET_DIR} after download.")
        sys.exit(1)

    archive_path = zip_files[0]
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"  Found archive: {archive_path.name} ({archive_size_mb:.1f} MB)")

    # Step 3: Extract archive
    print(f"\n[4/5] Extracting archive {archive_path.name}...")
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print("  Extraction complete.")

    # Remove the zip file to save disk space
    try:
        archive_path.unlink()
        print(f"  Cleaned up archive {archive_path.name}")
    except Exception as e:
        print(f"  Notice: Could not delete archive: {e}")

    # Step 4: Inspect extracted directory and reorganize splits if needed
    print("\n[5/5] Programmatically inspecting directories and image counts...")
    organize_dataset_splits()

def organize_dataset_splits():
    """
    Scans the extracted folders, counts class images programmatically,
    and ensures a clean train/val/test structure:
    datasets/fresh_stale/
      ├── train/
      │     ├── fresh/
      │     └── stale/
      ├── val/
      │     ├── fresh/
      │     └── stale/
      └── test/
            ├── fresh/
            └── stale/
    Also maintains category-level tags for the 6 food types.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Find all images
    all_image_paths = [p for p in DATASET_DIR.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]
    print(f"  Total raw images found: {len(all_image_paths)}")

    # Classify by freshness and fruit/veg type based on folder/file names
    class_stats = {
        "fresh": 0,
        "stale": 0
    }
    food_stats = {ft: {"fresh": 0, "stale": 0} for ft in SUPPORTED_FOOD_TYPES}

    structured_data = {"train": {"fresh": [], "stale": []},
                       "val": {"fresh": [], "stale": []},
                       "test": {"fresh": [], "stale": []}}

    # Check if the dataset already contains train/test split folders
    train_subdirs = [p for p in DATASET_DIR.iterdir() if p.is_dir() and "train" in p.name.lower()]
    test_subdirs = [p for p in DATASET_DIR.iterdir() if p.is_dir() and "test" in p.name.lower()]

    print(f"  Detected subdirectories: {[p.name for p in DATASET_DIR.iterdir() if p.is_dir()]}")

    # Inspect images and determine label
    categorized_images = {"fresh": [], "stale": []}
    corrupt_count = 0

    from PIL import Image

    for img_path in all_image_paths:
        # Verify file is readable / not corrupt
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception:
            corrupt_count += 1
            continue

        # Check immediate parent folder name (e.g. fresh_apple, stale_banana)
        parent_folder = img_path.parent.name.lower()
        if parent_folder.startswith("fresh"):
            label = "fresh"
        elif parent_folder.startswith("stale") or parent_folder.startswith("rotten"):
            label = "stale"
        else:
            # Fallback check file stem
            if "fresh" in img_path.stem.lower():
                label = "fresh"
            elif "stale" in img_path.stem.lower() or "rotten" in img_path.stem.lower():
                label = "stale"
            else:
                continue

        class_stats[label] += 1
        categorized_images[label].append(img_path)

        for ft in SUPPORTED_FOOD_TYPES:
            clean_ft = ft.replace(" ", "_")
            if clean_ft in parent_folder or ft in parent_folder:
                food_stats[ft][label] += 1
                break

    print(f"\n  Image Counts by Condition:")
    print(f"    Fresh images: {class_stats['fresh']}")
    print(f"    Stale images: {class_stats['stale']}")
    print(f"    Corrupt/Unreadable images: {corrupt_count}")

    print("\n  Food Types Distribution:")
    for ft, cnts in food_stats.items():
        print(f"    {ft.capitalize()}: {cnts['fresh']} fresh, {cnts['stale']} stale (Total: {cnts['fresh'] + cnts['stale']})")

    # Ensure train/val/test split exists (80% train, 10% val, 10% test)
    # Check if a standardized split dir already exists
    processed_dir = DATASET_DIR / "processed"
    if not processed_dir.exists():
        print("\n  Creating reproducible 80/10/10 train/val/test split in 'datasets/fresh_stale/processed/'...")
        processed_dir.mkdir(parents=True, exist_ok=True)
        random.seed(42)

        for label in ["fresh", "stale"]:
            imgs = list(categorized_images[label])
            random.shuffle(imgs)
            n_total = len(imgs)
            n_train = int(n_total * 0.8)
            n_val = int(n_total * 0.1)

            splits = {
                "train": imgs[:n_train],
                "val": imgs[n_train:n_train + n_val],
                "test": imgs[n_train + n_val:]
            }

            for split_name, split_imgs in splits.items():
                target_folder = processed_dir / split_name / label
                target_folder.mkdir(parents=True, exist_ok=True)
                for img_src in split_imgs:
                    target_file = target_folder / img_src.name
                    # Handle duplicate filenames across subfolders
                    if target_file.exists():
                        target_file = target_folder / f"{img_src.parent.name}_{img_src.name}"
                    shutil.copy2(img_src, target_file)

        print("  Split successfully created:")
        for sp in ["train", "val", "test"]:
            f_count = len(list((processed_dir / sp / "fresh").glob("*")))
            s_count = len(list((processed_dir / sp / "stale").glob("*")))
            print(f"    {sp.capitalize()}: {f_count} fresh, {s_count} stale (Total: {f_count + s_count})")

    print("="*70)
    print("SUCCESS: Fresh & Stale dataset successfully downloaded and verified.")
    print("="*70 + "\n")

if __name__ == "__main__":
    download_and_extract_freshness_dataset()
