"""
Module: ml/freshness/evaluate.py
Purpose: Standalone evaluation of trained freshness classifier on test split
"""

import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.freshness.dataset import FreshnessDataset, discover_freshness_samples, get_freshness_transforms
from ml.freshness.model import build_freshness_model

MODEL_PATH = PROJECT_ROOT / "models" / "freshness_best.pth"
DATASET_DIR = PROJECT_ROOT / "datasets" / "fresh_stale"
EVAL_DIR = PROJECT_ROOT / "ml" / "freshness" / "evaluation"

def evaluate_freshness_model():
    print("="*70)
    print("FRIDGE RESCUE — FRESHNESS MODEL STANDALONE TEST EVALUATION")
    print(f"Model path: {MODEL_PATH}")
    print("="*70)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model checkpoint not found at {MODEL_PATH}")
        print("Run python ml/freshness/train.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    arch = checkpoint.get("architecture", "mobilenet_v3_small")

    model = build_freshness_model(architecture=arch, pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_samples = discover_freshness_samples(DATASET_DIR, split="test")
    if not test_samples:
        test_samples = discover_freshness_samples(DATASET_DIR, split="val")

    print(f"Evaluating on {len(test_samples)} test samples...")
    _, eval_tf = get_freshness_transforms(img_size=224)
    test_dataset = FreshnessDataset(test_samples, transform=eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    print("\nTest Evaluation Results:")
    print(f"  Accuracy:         {acc*100:.2f}%")
    print(f"  Precision:        {prec*100:.2f}%")
    print(f"  Recall:           {rec*100:.2f}%")
    print(f"  F1 Score:         {f1*100:.2f}%")
    print(f"  Confusion Matrix: {cm}")

    out_file = EVAL_DIR / "standalone_test_evaluation.json"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm,
            "test_samples": len(test_samples)
        }, f, indent=2)
    print(f"Saved evaluation metrics to: {out_file}")
    print("="*70)

if __name__ == "__main__":
    evaluate_freshness_model()
