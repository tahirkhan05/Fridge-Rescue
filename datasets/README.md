# Fridge Rescue Datasets

> **Mandatory Dataset Notice**: Both datasets are mandatory components of Fridge Rescue. Dataset 1 is used for food-object detection and Dataset 2 is used for freshness classification.

---

## Dataset 1 — Smart Refrigerator (Food Object Detection)

- **Name**: Smart Refrigerator — Northumbria University Newcastle
- **Source**: Roboflow Universe
- **URL**: [https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr](https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr)
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Purpose**: Primary food-object detection dataset to locate food items inside domestic refrigerators.
- **Format**: YOLOv8 / YOLO11 format (`data.yaml`, `images/`, `labels/`).
- **Approximate Size**: ~3,049 images, 30 object-detection classes.
- **Classes**:
  - `apple`, `banana`, `tomato`, `carrot`, `chicken`, `potato`, `milk`, `bread`, `onion`, `cheese`, `corn`, `shrimp`, `butter`, `chocolate`, `beef`, `eggs`, `flour`, `mushrooms`, `strawberries`, `spinach`, `sweet potato`, `cucumber`, `bell pepper`, `yogurt`, `juice`, `sausage`, `cabbage`, `lettuce`, `ham`, `pasta`
- **Expected Directory**: `datasets/smart_refrigerator/`
- **Download Command**:
  ```bash
  python scripts/download_detection_dataset.py
  ```

---

## Dataset 2 — Fresh and Stale Images of Fruits and Vegetables (Freshness Classification)

- **Name**: Fresh and Stale Images of Fruits and Vegetables
- **Author**: Raghav R Potdar
- **Source**: Kaggle Datasets
- **URL**: [https://www.kaggle.com/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables](https://www.kaggle.com/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)
- **License**: CC0 Public Domain
- **Purpose**: Mandatory freshness-classification dataset to assess deterioration/spoilage state of detected produce items.
- **Approximate Size**: ~14.7k images, ~1.53 GB.
- **Food Types & Classes**:
  - 6 Produce Categories: `apple`, `banana`, `bitter gourd`, `capsicum`, `orange`, `tomato`
  - Binary Condition Labels: `fresh` vs `stale` (e.g., `freshapples`, `staleapples`, `freshtomato`, `staletomato`, etc.)
- **Expected Directory**: `datasets/fresh_stale/`
- **Download Command**:
  ```bash
  python scripts/download_freshness_dataset.py
  ```

---

## Reproducible Pipeline Verification

To programmatically verify image counts, class distribution, corrupt image checks, and train/val/test splits across both datasets, execute:

```bash
python scripts/verify_datasets.py
```
