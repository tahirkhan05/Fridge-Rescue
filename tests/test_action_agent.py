"""
Test Suite: tests/test_action_agent.py
Purpose: Test heuristic rescue agent fallback and recipe generation
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.priority.rescue_engine import RescuePriorityEngine
from ml.action_agent.rescue_agent import RescueActionAgent

def test_action_agent():
    agent = RescueActionAgent()

    # Scenario: Fridge has declining tomatoes, eggs, cheese
    items = [
        {"id": "1", "name": "tomato", "quantity": 4, "freshness": "stale", "freshness_confidence": 0.90},
        {"id": "2", "name": "eggs", "quantity": 6, "freshness": None},
        {"id": "3", "name": "cheese", "quantity": 1, "freshness": None}
    ]

    ranked = RescuePriorityEngine.rank_fridge_inventory(items)
    plan = agent.formulate_rescue_plan(ranked)

    assert plan is not None
    assert "headline" in plan
    assert len(plan["recommended_recipes"]) > 0
    rec = plan["recommended_recipes"][0]
    print(f"PASS: Action Agent generated: '{plan['headline']}'")
    print(f"      Recipe Title: {rec['title']}")
    print(f"      Rescued: {rec['priority_rescued']}")
    assert "tomato" in str(rec["ingredients_used"]).lower()

if __name__ == "__main__":
    test_action_agent()
    print("All Action Agent tests passed!")
