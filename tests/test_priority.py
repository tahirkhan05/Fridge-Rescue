"""
Test Suite: tests/test_priority.py
Purpose: Test explainable multi-signal Rescue Priority Engine
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.priority.rescue_engine import RescuePriorityEngine, FOOD_SAFETY_DISCLAIMER

def test_rescue_priority():
    # Case 1: Stale tomatoes with surplus -> Should be RESCUE NOW / HIGH
    stale_tomato = RescuePriorityEngine.calculate_priority(
        item_id="item_01",
        name="tomato",
        quantity=4,
        detection_confidence=0.95,
        freshness="stale",
        freshness_confidence=0.88,
        expiry_date=None
    )
    assert stale_tomato["tier"] == "RESCUE NOW", f"Expected RESCUE NOW, got {stale_tomato['tier']}"
    assert stale_tomato["priority"] == "HIGH"
    assert len(stale_tomato["reasons"]) >= 2
    assert "declining" in stale_tomato["reasons"][0].lower()
    print("PASS: Stale tomato correctly scored as RESCUE NOW (Score:", stale_tomato["score"], ")")

    # Case 2: Packaged milk expiring in 2 days -> Should be USE SOON / MEDIUM
    milk = RescuePriorityEngine.calculate_priority(
        item_id="item_02",
        name="milk",
        quantity=1,
        detection_confidence=0.91,
        freshness=None,
        expiry_date="2026-10-04",
        days_remaining=2
    )
    assert milk["priority"] in ["MEDIUM", "HIGH"]
    assert any("2 day" in r for r in milk["reasons"])
    print("PASS: Milk expiring in 2 days correctly prioritized (Score:", milk["score"], ")")

    # Case 3: Fresh apples -> Should be STABLE / LOW
    fresh_apple = RescuePriorityEngine.calculate_priority(
        item_id="item_03",
        name="apple",
        quantity=2,
        detection_confidence=0.93,
        freshness="fresh",
        freshness_confidence=0.94,
        expiry_date=None
    )
    assert fresh_apple["tier"] == "STABLE"
    assert fresh_apple["priority"] == "LOW"
    print("PASS: Fresh apples correctly scored as STABLE (Score:", fresh_apple["score"], ")")

    # Check disclaimer exists
    assert FOOD_SAFETY_DISCLAIMER in stale_tomato["food_safety_disclaimer"]
    print("PASS: Food safety disclaimer properly attached.")

if __name__ == "__main__":
    test_rescue_priority()
    print("All Rescue Priority Engine tests passed!")
