"""
Module: ml/detection/evaluate.py
Purpose: Independent evaluation script for trained YOLO food detection model on test dataset
"""

import json
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "detection_best.pt"
DATA_YAML = PROJECT_ROOT / "datasets" / "smart_refrigerator" / "data.yaml"
EVAL_DIR = PROJECT_ROOT / "ml" / "detection" / "evaluation"

def evaluate_model():
    print("="*70)
    print("FRIDGE RESCUE — FOOD DETECTION MODEL INDEPENDENT EVALUATION")
    print(f"Model path: {MODEL_PATH}")
    print(f"Data config: {DATA_YAML}")
    print("="*70)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Train it first using python ml/detection/train.py")
        return

    model = YOLO(str(MODEL_PATH))
    val_results = model.val(data=str(DATA_YAML), split="test")

    metrics_dict = {
        "precision": float(val_results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(val_results.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "fitness": float(val_results.results_dict.get("fitness", 0.0))
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EVAL_DIR / "test_evaluation_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print("\nRecorded Test Set Metrics:")
    print(f"  Precision: {metrics_dict['precision']:.4f}")
    print(f"  Recall:    {metrics_dict['recall']:.4f}")
    print(f"  mAP@50:    {metrics_dict['mAP50']:.4f}")
    print(f"  mAP@50:95: {metrics_dict['mAP50_95']:.4f}")
    print(f"\nSaved metrics to: {out_file}")
    print("="*70)

if __name__ == "__main__":
    evaluate_model()
