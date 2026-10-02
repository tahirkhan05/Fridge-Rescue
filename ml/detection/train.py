"""
Module: ml/detection/train.py
Purpose: Fine-tune YOLO11n on the mandatory Smart Refrigerator dataset
Input: datasets/smart_refrigerator/data.yaml
Output: models/detection_best.pt, ml/detection/evaluation/metrics.json
"""

import os
import sys
import json
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "smart_refrigerator"
DATA_YAML = DATASET_DIR / "data.yaml"
OUTPUT_MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "ml" / "detection" / "evaluation"

def prepare_data_yaml():
    """
    Ensures data.yaml paths resolve correctly for Ultralytics YOLO.
    """
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}. Run scripts/download_detection_dataset.py first.")

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set absolute base path to dataset dir
    config["path"] = str(DATASET_DIR.resolve())
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    print(f"Verified and updated {DATA_YAML} with path: {config['path']}")
    return DATA_YAML

def train_detection_model(epochs: int = 15, batch_size: int = 16, img_size: int = 640):
    print("="*70)
    print("FRIDGE RESCUE — MODEL 1: FOOD DETECTION TRAINING (YOLO11)")
    print("Dataset: Smart Refrigerator — Northumbria University Newcastle")
    print(f"Device: {'CUDA GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)

    prepare_data_yaml()

    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Check device
    device = 0 if torch.cuda.is_available() else "cpu"

    print("\n[1/4] Initializing YOLO11n base model...")
    try:
        model = YOLO("yolo11n.pt")
    except Exception as e:
        print(f"Notice: Failed to load yolo11n.pt ({e}), using yolov8n.pt...")
        model = YOLO("yolov8n.pt")

    print(f"\n[2/4] Starting training for {epochs} epochs on {device}...")
    run_name = "smart_fridge_yolo"
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        verbose=True
    )

    print("\n[3/4] Validating trained model on test/val set...")
    val_metrics = model.val(data=str(DATA_YAML), split="test")

    # Extract real metrics
    metrics_dict = {
        "model_architecture": "YOLO11n",
        "dataset": "Smart Refrigerator — Northumbria University Newcastle",
        "num_classes": len(model.names),
        "classes": model.names,
        "epochs_trained": epochs,
        "precision": float(val_metrics.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(val_metrics.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "fitness": float(val_metrics.results_dict.get("fitness", 0.0))
    }

    metrics_file = EVAL_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Real metrics recorded to: {metrics_file}")
    print(f"  Precision: {metrics_dict['precision']:.4f}")
    print(f"  Recall:    {metrics_dict['recall']:.4f}")
    print(f"  mAP@50:    {metrics_dict['mAP50']:.4f}")
    print(f"  mAP@50:95: {metrics_dict['mAP50_95']:.4f}")

    # Copy best model to models/
    run_dir = Path(model.trainer.save_dir)
    best_pt = run_dir / "weights" / "best.pt"
    target_best_pt = OUTPUT_MODELS_DIR / "detection_best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, target_best_pt)
        print(f"\n[4/4] Best trained weights exported to: {target_best_pt}")
    else:
        last_pt = run_dir / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, target_best_pt)
            print(f"\n[4/4] Exported last trained weights to: {target_best_pt}")

    # Copy plots & confusion matrix to evaluation folder
    for plot_file in run_dir.glob("*.png"):
        shutil.copy2(plot_file, EVAL_DIR / plot_file.name)
    print(f"Evaluation plots and confusion matrices archived in: {EVAL_DIR}")
    print("="*70)
    print("TRAINING & EXPORT COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train_detection_model(epochs=args.epochs, batch_size=args.batch, img_size=args.imgsz)
