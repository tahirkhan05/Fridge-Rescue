"""
Module: ml/priority/rescue_engine.py
Purpose: Explainable Multi-Signal Rescue Priority Decision Engine
Combines:
  1. Visual freshness classifier predictions (fresh vs stale)
  2. OCR-extracted expiry date proximity (days remaining)
  3. Intrinsic item perishability classification
  4. Surplus quantity impact
  5. Personal user adjustments and storage context
"""

import datetime
from typing import Dict, Any, List, Optional

FOOD_SAFETY_DISCLAIMER = (
    "Fridge Rescue provides AI-assisted food-waste guidance, not a food-safety guarantee. "
    "Image classification cannot guarantee microbiological safety. "
    "Always verify packaging dates, smell, texture, and food condition before consuming."
)

# Baseline perishability decay speeds (in days under domestic refrigeration)
PERISHABILITY_TIERS = {
    # High perishability: fast spoilage (3-5 days)
    "high": {
        "berries", "strawberries", "spinach", "mushrooms", "fresh fish", 
        "chicken", "beef", "shrimp", "lettuce", "opened milk"
    },
    # Moderate perishability: (5-10 days)
    "moderate": {
        "tomato", "banana", "capsicum", "cucumber", "bitter gourd",
        "milk", "yogurt", "cheese", "bread", "ham", "sausage"
    },
    # Low perishability: shelf-stable or durable (10-30+ days)
    "low": {
        "apple", "orange", "potato", "sweet potato", "onion", "carrot",
        "cabbage", "butter", "eggs", "chocolate", "flour", "pasta", "corn"
    }
}

class RescuePriorityEngine:
    """
    Computes a transparent 0-100 Rescue Priority Score and generates
    human-readable, explainable rationales for every item.
    """

    @classmethod
    def get_perishability(cls, item_name: str) -> str:
        name_lower = item_name.lower()
        for item in PERISHABILITY_TIERS["high"]:
            if item in name_lower:
                return "high"
        for item in PERISHABILITY_TIERS["moderate"]:
            if item in name_lower:
                return "moderate"
        for item in PERISHABILITY_TIERS["low"]:
            if item in name_lower:
                return "low"
        return "moderate"

    @classmethod
    def calculate_priority(
        cls,
        item_id: str,
        name: str,
        quantity: int,
        detection_confidence: float,
        freshness: Optional[str] = None,              # 'fresh', 'stale', or None
        freshness_confidence: Optional[float] = None, # 0.0 to 1.0
        expiry_date: Optional[str] = None,            # 'YYYY-MM-DD'
        days_remaining: Optional[int] = None,
        user_adjusted: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates item score (0-100), urgency tier (RESCUE NOW, USE SOON, STABLE),
        and list of bullet-point explanatory reasons.
        """
        score = 0.0
        reasons = []

        # 1. Freshness Signal (Weight up to 45 pts)
        if freshness is not None and freshness.lower() == "stale":
            f_conf = freshness_confidence if freshness_confidence is not None else 0.85
            stale_pts = 35 + (f_conf * 10)  # 35 to 45 pts
            score += stale_pts
            reasons.append(
                f"Freshness classifier indicates declining visual condition ({int(f_conf*100)}% confidence)"
            )
        elif freshness is not None and freshness.lower() == "fresh":
            f_conf = freshness_confidence if freshness_confidence is not None else 0.85
            reasons.append(f"Visual condition appears fresh ({int(f_conf*100)}% confidence)")
        else:
            reasons.append("Produce visual freshness assessment unavailable for this item class")

        # 2. Expiry Proximity Signal (Weight up to 50 pts)
        if expiry_date:
            if days_remaining is None:
                try:
                    exp_dt = datetime.datetime.strptime(expiry_date, "%Y-%m-%d").date()
                    today = datetime.date.today()
                    days_remaining = (exp_dt - today).days
                except Exception:
                    pass

            if days_remaining is not None:
                if days_remaining < 0:
                    score += 50
                    reasons.append(f"Package date was {abs(days_remaining)} day(s) ago ({expiry_date}) — urgent check required")
                elif days_remaining == 0:
                    score += 45
                    reasons.append(f"Package date is TODAY ({expiry_date})")
                elif days_remaining <= 2:
                    score += 35
                    reasons.append(f"Package date in {days_remaining} day(s) ({expiry_date})")
                elif days_remaining <= 5:
                    score += 20
                    reasons.append(f"Package date approaching in {days_remaining} days ({expiry_date})")
                else:
                    reasons.append(f"Package date has {days_remaining} days remaining ({expiry_date})")

        # 3. Perishability Baseline (Weight up to 20 pts)
        perish_tier = cls.get_perishability(name)
        if perish_tier == "high":
            score += 20
            reasons.append(f"Category '{name}' is inherently highly perishable")
        elif perish_tier == "moderate":
            score += 10
            reasons.append(f"Category '{name}' has moderate refrigerator shelf-life")
        else:
            reasons.append(f"Category '{name}' is durable / relatively shelf-stable")

        # 4. Quantity Surplus Impact (Weight up to 10 pts)
        if quantity >= 4:
            score += 10
            reasons.append(f"Surplus volume ({quantity} units detected) increases waste risk if unused")
        elif quantity >= 2:
            score += 5

        # Normalize score bounds
        final_score = max(0, min(100, int(round(score))))

        # Determine Tier
        if final_score >= 60:
            priority_label = "HIGH"
            tier = "RESCUE NOW"
            action_tag = "Urgent: Cook or freeze today"
        elif final_score >= 35:
            priority_label = "MEDIUM"
            tier = "USE SOON"
            action_tag = "Plan to use in next 2-3 days"
        else:
            priority_label = "LOW"
            tier = "STABLE"
            action_tag = "Stable condition; safe for later this week"

        return {
            "id": item_id,
            "name": name,
            "quantity": quantity,
            "detection_confidence": round(detection_confidence, 2),
            "freshness": freshness,
            "freshness_confidence": round(freshness_confidence, 2) if freshness_confidence is not None else None,
            "expiry_date": expiry_date,
            "days_remaining": days_remaining,
            "perishability": perish_tier,
            "priority": priority_label,
            "tier": tier,
            "score": final_score,
            "action_tag": action_tag,
            "reasons": reasons,
            "user_adjusted": user_adjusted,
            "food_safety_disclaimer": FOOD_SAFETY_DISCLAIMER
        }

    @classmethod
    def rank_fridge_inventory(cls, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ranks all items by priority score and categorizes into RESCUE NOW, USE SOON, STABLE.
        """
        scored_items = []
        for it in items:
            scored = cls.calculate_priority(
                item_id=it.get("id", f"item_{len(scored_items)+1}"),
                name=it.get("name", "Unknown item"),
                quantity=it.get("quantity", 1),
                detection_confidence=it.get("detection_confidence", 0.9),
                freshness=it.get("freshness"),
                freshness_confidence=it.get("freshness_confidence"),
                expiry_date=it.get("expiry_date"),
                days_remaining=it.get("days_remaining"),
                user_adjusted=it.get("user_adjusted", False)
            )
            scored_items.append(scored)

        # Sort descending by priority score
        scored_items.sort(key=lambda x: x["score"], reverse=True)

        rescue_now = [it for it in scored_items if it["tier"] == "RESCUE NOW"]
        use_soon = [it for it in scored_items if it["tier"] == "USE SOON"]
        stable = [it for it in scored_items if it["tier"] == "STABLE"]

        return {
            "total_items": sum(it["quantity"] for it in scored_items),
            "total_unique_items": len(scored_items),
            "rescue_now_count": len(rescue_now),
            "use_soon_count": len(use_soon),
            "stable_count": len(stable),
            "items": scored_items,
            "grouped": {
                "rescue_now": rescue_now,
                "use_soon": use_soon,
                "stable": stable
            },
            "disclaimer": FOOD_SAFETY_DISCLAIMER
        }
