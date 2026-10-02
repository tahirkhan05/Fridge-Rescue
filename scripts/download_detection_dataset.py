"""
Script: scripts/download_detection_dataset.py
Purpose: Download the mandatory Smart Refrigerator dataset from Roboflow Universe
Dataset: Smart Refrigerator — Northumbria University Newcastle
Source: https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr
License: CC BY 4.0
Target location: datasets/smart_refrigerator/
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "smart_refrigerator"

def download_smart_refrigerator_dataset():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("\n" + "="*70)
        print("ERROR: ROBOFLOW_API_KEY not found in environment or .env file.")
        print("Please obtain an API key from https://app.roboflow.com/ and set it:")
        print("  1. Add ROBOFLOW_API_KEY=your_key to your .env file")
        print("  2. Or run: export ROBOFLOW_API_KEY=your_key (Linux/Mac)")
        print("     or: $env:ROBOFLOW_API_KEY='your_key' (Windows PowerShell)")
        print("="*70 + "\n")
        sys.exit(1)

    print("="*70)
    print("FRIDGE RESCUE — DATASET 1: SMART REFRIGERATOR DOWNLOAD")
    print("Source: https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr")
    print("License: CC BY 4.0")
    print(f"Destination: {DATASET_DIR}")
    print("="*70)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: 'roboflow' package not installed. Run: pip install roboflow")
        sys.exit(1)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace("northumbria-university-newcastle")
    project = workspace.project("smart-refrigerator-zryjr")
    
    # Roboflow version download
    version = project.version(2)
    print(f"\n[1/3] Downloading dataset version {version.version} (format: yolov11)...")
    
    res = version.download("yolov11")
    download_loc = Path(getattr(res, 'location', 'smart-refrigerator-2'))
    if download_loc.resolve() != DATASET_DIR.resolve() and download_loc.exists():
        import shutil
        for item in download_loc.iterdir():
            dest = DATASET_DIR / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        try:
            shutil.rmtree(download_loc)
        except Exception:
            pass

    print(f"[2/3] Download complete to: {DATASET_DIR}")

    # Inspect and verify structure
    print("\n[3/3] Inspecting downloaded dataset...")
    data_yaml_path = DATASET_DIR / "data.yaml"
    if not data_yaml_path.exists():
        print(f"Warning: data.yaml not found at {data_yaml_path}. Checking subdirectories...")
        found_yamls = list(DATASET_DIR.rglob("data.yaml"))
        if found_yamls:
            data_yaml_path = found_yamls[0]

    if data_yaml_path.exists():
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        classes = meta.get("names", [])
        num_classes = meta.get("nc", len(classes))
        print(f"  Configuration file: {data_yaml_path}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Sample classes: {classes[:10]}...")
    
    splits = ["train", "valid", "test"]
    split_counts = {}
    for s in splits:
        img_dir = DATASET_DIR / s / "images"
        if not img_dir.exists():
            # Check alternative Roboflow layout
            img_dir = DATASET_DIR / s
        if img_dir.exists():
            count = len([f for f in img_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            split_counts[s] = count
            print(f"  {s.capitalize()} images: {count}")
        else:
            split_counts[s] = 0

    total_images = sum(split_counts.values())
    print(f"  Total images verified: {total_images}")
    print("="*70)
    print("SUCCESS: Smart Refrigerator dataset successfully downloaded and verified.")
    print("="*70 + "\n")

if __name__ == "__main__":
    download_smart_refrigerator_dataset()
