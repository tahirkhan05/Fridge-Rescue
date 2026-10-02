"""
Module: ml/freshness/train.py
Purpose: Train transfer learning model on Fresh and Stale fruits/vegetables dataset
Outputs: models/freshness_best.pth, ml/freshness/evaluation/metrics.json
"""

import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.freshness.dataset import FreshnessDataset, discover_freshness_samples, get_freshness_transforms
from ml.freshness.model import build_freshness_model

DATASET_DIR = PROJECT_ROOT / "datasets" / "fresh_stale"
MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "ml" / "freshness" / "evaluation"

def train_freshness_model(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.0005,
    architecture: str = "mobilenet_v3_small",
    max_samples_per_split: int = None
):
    print("="*70)
    print("FRIDGE RESCUE — MODEL 2: FRESHNESS CLASSIFIER TRAINING")
    print("Dataset: Fresh and Stale Images of Fruits and Vegetables (Kaggle)")
    print(f"Architecture: {architecture}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("="*70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover data samples
    train_samples = discover_freshness_samples(DATASET_DIR, split="train")
    val_samples = discover_freshness_samples(DATASET_DIR, split="val")
    test_samples = discover_freshness_samples(DATASET_DIR, split="test")

    if not train_samples:
        print(f"ERROR: No training samples found in {DATASET_DIR}.")
        print("Run: python scripts/download_freshness_dataset.py first.")
        sys.exit(1)

    # Subsample if max_samples_per_split is specified
    if max_samples_per_split:
        import random
        random.seed(42)
        random.shuffle(train_samples)
        train_samples = train_samples[:max_samples_per_split]
        random.shuffle(val_samples)
        val_samples = val_samples[:max(100, max_samples_per_split // 5)]

    print(f"Loaded samples:")
    print(f"  Train: {len(train_samples)} images")
    print(f"  Val:   {len(val_samples)} images")
    print(f"  Test:  {len(test_samples)} images")

    # 2. Setup transforms and data loaders (num_workers=0 to prevent Windows IPC bottlenecks)
    train_tf, eval_tf = get_freshness_transforms(img_size=224)
    train_dataset = FreshnessDataset(train_samples, transform=train_tf)
    val_dataset = FreshnessDataset(val_samples if val_samples else test_samples, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Model, Criterion, Optimizer
    model = build_freshness_model(architecture=architecture, pretrained=True, num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = MODELS_DIR / "freshness_best.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\nStarting Training Loop...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        val_loss = val_running_loss / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": architecture,
                "classes": ["fresh", "stale"],
                "best_val_acc": best_val_acc,
                "epoch": epoch
            }, best_weights_path)
            print(f"  --> Saved new best checkpoint to {best_weights_path} (Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes. Best Val Acc: {best_val_acc*100:.2f}%")

    # 4. Evaluate on Test Set
    print("\nRunning Final Test Evaluation...")
    eval_target_samples = test_samples if test_samples else val_samples
    test_dataset = FreshnessDataset(eval_target_samples, transform=eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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

    metrics_dict = {
        "architecture": architecture,
        "dataset": "Fresh and Stale Images of Fruits and Vegetables (Kaggle)",
        "classes": ["fresh", "stale"],
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "train_samples": len(train_samples),
        "test_samples": len(eval_target_samples),
        "epochs": epochs
    }

    metrics_file = EVAL_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  Confusion Matrix: {cm}")

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Freshness Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Freshness Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    curve_plot = EVAL_DIR / "training_curves.png"
    plt.savefig(curve_plot, dpi=120)
    plt.close()
    print(f"Training curves saved to: {curve_plot}")

    print("="*70)
    print("FRESHNESS MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    train_freshness_model(epochs=args.epochs, batch_size=args.batch, architecture=args.arch, max_samples_per_split=args.max_samples)
