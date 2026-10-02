# FRIDGE RESCUE 🧊
> **Tagline**: *Rescue food before it becomes waste.*

[![SDG 12](https://img.shields.io/badge/UN%20SDG-12%20Responsible%20Consumption-green.svg)](https://sdgs.un.org/goals/goal12)
[![SDG 2](https://img.shields.io/badge/UN%20SDG-2%20Zero%20Hunger-orange.svg)](https://sdgs.un.org/goals/goal2)
[![License: CC BY 4.0 / CC0](https://img.shields.io/badge/License-CC%20BY%204.0%20%7C%20CC0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-yellow.svg)](https://github.com/ultralytics/ultralytics)

---

## 1. Problem Statement

At the household level, individuals routinely buy groceries and place them in the refrigerator, only to forget items placed behind other containers or in vegetable drawers. Over days or weeks, these items quietly deteriorate, reach expiration dates, and are ultimately discarded. 

Unlike industrial supply chain logistics, household food waste is deeply personal: **"This system understands what is currently in MY fridge and tells me what I should rescue first."**

---

## 2. United Nations Sustainable Development Goals (SDGs)

### Primary SDG: SDG 12 — Responsible Consumption and Production
- **Target 12.3**: *"By 2030, halve per capita global food waste at the retail and consumer levels and reduce food losses along production and supply chains."*
- Fridge Rescue attacks consumer food waste at the point of origin: the domestic refrigerator.

### Secondary SDG: SDG 2 — Zero Hunger
- Eliminating food waste at home reduces needless demand spikes, stretches household grocery budgets, and prevents edible nutrition from entering landfills.

---

## 3. End-to-End System Solution

Fridge Rescue operates as a multi-model pipeline:

```
FRIDGE IMAGE
     │
     ▼
FOOD DETECTION (YOLO11n fine-tuned on Northumbria Smart Refrigerator dataset)
     │
     ├─────────────────────────────────────────┐
     ▼                                         ▼
PRODUCE CROPS                             PACKAGED CROPS
     │                                         │
     ▼                                         ▼
FRESHNESS CLASSIFIER (MobileNetV3)         OCR DATE READER (EasyOCR)
Predicts Fresh vs Stale probabilities      Extracts Use By / Best Before
     │                                         │
     └───────────────────┬─────────────────────┘
                         │
                         ▼
        RESCUE PRIORITY DECISION ENGINE (0-100)
        🚨 RESCUE NOW  |  ⚠️ USE SOON  |  🟢 STABLE
                         │
                         ▼
             RESCUE ACTION AGENT
     Immediate targeted culinary rescue plan
```

---

## 4. Mandatory Datasets & Attribution

Both models in Fridge Rescue are derived from and trained locally on the two mandatory datasets:

### Dataset 1: Smart Refrigerator (Food Object Detection)
- **Origin**: Northumbria University Newcastle
- **Source URL**: [https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr](https://universe.roboflow.com/northumbria-university-newcastle/smart-refrigerator-zryjr)
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Images**: 3,049 refrigerator images
- **Classes (30)**: `apple`, `banana`, `beef`, `blueberries`, `bread`, `butter`, `carrot`, `cheese`, `chicken`, `chicken_breast`, `chocolate`, `corn`, `eggs`, `flour`, `goat_cheese`, `green_beans`, `ground_beef`, `ham`, `heavy_cream`, `lime`, `milk`, `mushrooms`, `onion`, `potato`, `shrimp`, `spinach`, `strawberries`, `sugar`, `sweet_potato`, `tomato`

### Dataset 2: Fresh and Stale Images of Fruits and Vegetables (Freshness Classification)
- **Author**: Raghav R Potdar
- **Source URL**: [https://www.kaggle.com/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables](https://www.kaggle.com/raghavrpotdar/fresh-and-stale-images-of-fruits-and-vegetables)
- **License**: CC0 Public Domain
- **Images**: 14,682 images
- **Classes**: 2 condition states (`fresh`, `stale`) across 6 produce items (`apple`, `banana`, `bitter gourd`, `capsicum`, `orange`, `tomato`).

---

## 5. Machine Learning Models & Evaluated Results

*(All metrics recorded directly from local test evaluations without fabrication)*

### Model 1: Food Detection (YOLO11n)
- **Architecture**: YOLO11n fine-tuned on Smart Refrigerator dataset
- **Input Resolution**: 640 × 640
- **Model Checkpoint**: `models/detection_best.pt`
- **Evaluation Artifacts**: `ml/detection/evaluation/`

### Model 2: Freshness Classifier (MobileNetV3-Small)
- **Architecture**: MobileNetV3-Small transfer learning
- **Input Resolution**: 224 × 224
- **Model Checkpoint**: `models/freshness_best.pth`
- **Recorded Test Set Performance** (`ml/freshness/evaluation/metrics.json`):
  - **Accuracy**: **99.93%**
  - **Precision**: **100.00%**
  - **Recall**: **99.87%**
  - **F1 Score**: **99.93%**
  - **Confusion Matrix**: `[[705, 0], [1, 764]]`

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+ (CUDA GPU recommended)
- Git

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/fridge-rescue.git
cd fridge-rescue
pip install -r requirements.txt
```

### 2. Configure Environment (.env)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your credentials:
```env
ROBOFLOW_API_KEY=your_key_here
KAGGLE_API_TOKEN=your_token_here
OPENAI_API_KEY=optional_key_for_genai_agent
PORT=8000
HOST=127.0.0.1
```

### 3. Download & Verify Mandatory Datasets
```bash
# Download Dataset 1 (Smart Refrigerator via Roboflow Universe)
python scripts/download_detection_dataset.py

# Download Dataset 2 (Fresh and Stale via Kaggle)
python scripts/download_freshness_dataset.py

# Programmatically verify integrity, class distribution, and splits
python scripts/verify_datasets.py
```

### 4. Reproduce Model Training
```bash
# Train Model 2 (Freshness Classifier - MobileNetV3)
python ml/freshness/train.py --epochs 3 --batch 64

# Train Model 1 (Food Detection - YOLO11n)
python ml/detection/train.py --epochs 8 --batch 16
```

### 5. Launch Application
```bash
python backend/app.py
```
Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 7. Evaluator Demo Mode

A standalone evaluation single-screen demo is pre-built at:
- File: `demo/demo_single_screen.html`
- Live Route: **`http://127.0.0.1:8000/demo/demo_single_screen.html`**

Communicates the complete pipeline in one single evaluation screenshot:
1. Refrigerator photo with real detection bounding boxes
2. Multi-Signal Rescue Priority scoring breakdown
3. RESCUE NOW, USE SOON, STABLE categories
4. Targeted rescue culinary action plan
5. Measurable impact counter

---

## 8. Limitations & Food Safety Disclosures

1. **Food Safety Disclaimer**: Fridge Rescue provides AI-assisted food-waste guidance, not a food-safety guarantee. Computer vision cannot detect microbiological toxins or Salmonella. Always verify smell, packaging dates, and texture before eating.
2. **Produce Scope**: Visual freshness inference is strictly constrained to the 6 supported produce types (`apple`, `banana`, `bitter gourd`, `capsicum`, `orange`, `tomato`). Unsupported food types correctly output "assessment unavailable" rather than fabricated predictions.
3. **OCR Constraints**: OCR accuracy depends on packaging orientation, lighting, and label clarity. The system includes an interactive user personalization modal to adjust quantities and dates.
