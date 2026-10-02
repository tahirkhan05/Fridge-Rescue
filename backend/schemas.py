"""
Module: backend/schemas.py
Purpose: Pydantic schemas for request/response serialization
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ItemUpdateRequest(BaseModel):
    name: Optional[str] = None
    quantity: Optional[int] = None
    expiry_date: Optional[str] = None
    freshness: Optional[str] = None
    rescued: Optional[int] = None

class FridgeItemSchema(BaseModel):
    id: str
    name: str
    quantity: int
    detection_confidence: float
    freshness: Optional[str] = None
    freshness_confidence: Optional[float] = None
    expiry_date: Optional[str] = None
    days_remaining: Optional[int] = None
    perishability: str
    priority: str
    tier: str
    score: int
    action_tag: str
    reasons: List[str]
    user_adjusted: bool = False
    bbox: Optional[List[float]] = None

class RankedInventoryResponse(BaseModel):
    total_items: int
    total_unique_items: int
    rescue_now_count: int
    use_soon_count: int
    stable_count: int
    items: List[Dict[str, Any]]
    grouped: Dict[str, List[Dict[str, Any]]]
    disclaimer: str

class RescuePlanResponse(BaseModel):
    headline: str
    summary: str
    target_rescued_items: List[str]
    recommended_recipes: List[Dict[str, Any]]
    engine_used: str
    impact_statement: str

class ScanResponse(BaseModel):
    scan_id: int
    annotated_image_url: str
    inventory: RankedInventoryResponse
    rescue_plan: RescuePlanResponse
