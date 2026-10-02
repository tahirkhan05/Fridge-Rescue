"""
Module: backend/app.py
Purpose: FastAPI Backend Server orchestrating the Fridge Rescue ML Pipeline
End-to-End Workflow:
  Image -> YOLO Food Detection -> Produce Crop Freshness -> OCR Expiry -> Rescue Priority -> GenAI Rescue Action
"""

import os
import sys
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from ml.detection.predict import FoodDetector
from ml.freshness.predict import FreshnessClassifier
from ml.ocr.expiry_detector import ExpiryDateDetector
from ml.priority.rescue_engine import RescuePriorityEngine, FOOD_SAFETY_DISCLAIMER
from ml.action_agent.rescue_agent import RescueActionAgent
from backend.database import save_scan, update_inventory_item, get_recent_scans, get_current_inventory
from backend.schemas import ItemUpdateRequest

# Initialize FastAPI
app = FastAPI(
    title="Fridge Rescue API",
    description="End-to-End AI System for Refrigerator Food-Waste Reduction (SDG 12 & SDG 2)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOADS_DIR = PROJECT_ROOT / "uploads"
ANNOTATED_DIR = PROJECT_ROOT / "uploads" / "annotated"
DEMO_DIR = PROJECT_ROOT / "demo"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/demo", StaticFiles(directory=str(DEMO_DIR)), name="demo")
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Lazy-loaded pipeline instances
_detector = None
_freshness = None
_ocr = None
_agent = None

def get_detector():
    global _detector
    if _detector is None:
        _detector = FoodDetector()
    return _detector

def get_freshness():
    global _freshness
    if _freshness is None:
        _freshness = FreshnessClassifier()
    return _freshness

def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = ExpiryDateDetector()
    return _ocr

def get_agent():
    global _agent
    if _agent is None:
        _agent = RescueActionAgent()
    return _agent

def process_fridge_image(image_path: Path) -> Dict[str, Any]:
    """
    Executes the 5-stage Fridge Rescue pipeline on an input image:
      1. Food Detection (YOLO11)
      2. Produce Freshness Classification (MobileNetV3)
      3. OCR Expiry Date Extraction (EasyOCR)
      4. Rescue Priority Engine (Multi-signal scoring)
      5. Rescue Action Agent (Smart recipe recommendations)
    """
    detector = get_detector()
    freshness_clf = get_freshness()
    ocr_detector = get_ocr()
    action_agent = get_agent()

    # Stage 1: Food Detection
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detections, annotated_rgb = detector.detect(img_rgb)

    # Save annotated visualization
    annotated_filename = f"annotated_{image_path.name}"
    annotated_path = ANNOTATED_DIR / annotated_filename
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(annotated_path), annotated_bgr)

    # Aggregate item quantities by detected class
    grouped_detections: Dict[str, List[Dict[str, Any]]] = {}
    for d in detections:
        c_name = d["name"]
        if c_name not in grouped_detections:
            grouped_detections[c_name] = []
        grouped_detections[c_name].append(d)

    raw_items = []
    item_counter = 1

    for class_name, dets in grouped_detections.items():
        qty = len(dets)
        best_det = max(dets, key=lambda x: x["confidence"])
        crop = best_det["crop"]

        # Stage 2: Freshness Classifier for produce items
        freshness_res = freshness_clf.predict_crop(crop, class_name)
        item_freshness = freshness_res.get("freshness")
        item_fresh_conf = freshness_res.get("freshness_confidence")

        # Stage 3: OCR Date Extraction (primarily packaged, dairy, meat, but applied to best crop)
        # Attempt OCR on crop or full region
        ocr_res = ocr_detector.extract_expiry(crop)
        expiry_date = ocr_res.get("expiry_date")
        days_rem = ocr_res.get("days_remaining")

        raw_items.append({
            "id": f"item_{item_counter:03d}",
            "name": class_name,
            "quantity": qty,
            "detection_confidence": best_det["confidence"],
            "freshness": item_freshness,
            "freshness_confidence": item_fresh_conf,
            "expiry_date": expiry_date,
            "days_remaining": days_rem,
            "bbox": best_det["bbox"],
            "user_adjusted": False
        })
        item_counter += 1

    # Stage 4: Multi-Signal Rescue Priority Ranking
    ranked_inventory = RescuePriorityEngine.rank_fridge_inventory(raw_items)

    # Stage 5: Rescue Action Agent
    rescue_plan = action_agent.formulate_rescue_plan(ranked_inventory)

    # Save to SQLite history
    scan_id = save_scan(
        image_path=f"/uploads/{image_path.name}",
        annotated_image_path=f"/uploads/annotated/{annotated_filename}",
        ranked_inventory=ranked_inventory,
        rescue_plan=rescue_plan
    )

    return {
        "scan_id": scan_id,
        "annotated_image_url": f"/uploads/annotated/{annotated_filename}",
        "raw_image_url": f"/uploads/{image_path.name}",
        "inventory": ranked_inventory,
        "rescue_plan": rescue_plan
    }

@app.get("/")
def serve_index():
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {
        "project": "Fridge Rescue",
        "tagline": "Rescue food before it becomes waste.",
        "sdg": "SDG 12 — Responsible Consumption and Production",
        "status": "API Server Active. Open /docs for Swagger UI or index.html in frontend."
    }

@app.post("/api/scan")
async def scan_fridge(image: UploadFile = File(...)):
    """
    Upload and analyze a refrigerator photo.
    """
    ext = Path(image.filename).suffix or ".jpg"
    filename = f"scan_{uuid.uuid4().hex[:10]}{ext}"
    target_path = UPLOADS_DIR / filename

    content = await image.read()
    with open(target_path, "wb") as f:
        f.write(content)

    try:
        result = process_fridge_image(target_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/demo/{demo_id}")
def run_demo_scan(demo_id: int = 1):
    """
    Runs a real scan on bundled demo fridge images.
    """
    demo_files = list(DEMO_DIR.glob("*.jpg")) + list(DEMO_DIR.glob("*.png"))
    if not demo_files:
        raise HTTPException(status_code=404, detail="No demo images found in demo/ directory.")

    selected_idx = (demo_id - 1) % len(demo_files)
    sample_file = demo_files[selected_idx]

    # Copy to uploads so it has a permanent scan record
    dest_name = f"demo_scan_{sample_file.name}"
    dest_path = UPLOADS_DIR / dest_name
    import shutil
    shutil.copy2(sample_file, dest_path)

    result = process_fridge_image(dest_path)
    return result

@app.get("/api/scans")
def list_scans():
    """
    Retrieve previous fridge scans from SQLite database.
    """
    return get_recent_scans()

@app.get("/api/inventory")
def list_inventory():
    """
    Retrieve current personal fridge inventory.
    """
    return get_current_inventory()

@app.patch("/api/inventory/{item_id}")
def update_item(item_id: str, payload: ItemUpdateRequest):
    """
    Allows personal user correction of quantity, expiry date, freshness, etc.
    """
    updates = {k: v for k, v in payload.dict().items() if v is not None}
    update_inventory_item(item_id, updates)
    return {"status": "success", "item_id": item_id, "updated": updates}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "disclaimer": FOOD_SAFETY_DISCLAIMER
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, ws="none")
