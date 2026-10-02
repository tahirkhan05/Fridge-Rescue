"""
Module: ml/action_agent/rescue_agent.py
Purpose: Generates food-waste reduction actions and smart rescue recipes
Features:
  - Generative AI agent (OpenAI API if key present)
  - Resilient Deterministic Fallback Engine (zero failure, offline-ready)
  - Prioritizes highest urgency items to maximize immediate food rescue
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Fallback Recipe Knowledge Base mapping key fridge surplus combinations to practical recipes
RESCUE_RECIPES_DB = [
    {
        "title": "Rustic Tomato & Herb Cheese Omelette",
        "key_ingredients": ["tomato", "eggs", "cheese"],
        "min_matches": 2,
        "prep_time_minutes": 10,
        "difficulty": "Easy",
        "steps": [
            "Dice tomatoes and gently sauté in a pan with butter or oil until softened (2-3 mins).",
            "Whisk eggs with a pinch of salt and black pepper; pour over softened tomatoes.",
            "Sprinkle shredded cheese on top; fold omelette when eggs set and cheese melts."
        ],
        "waste_reduction_tip": "Soft or overripe tomatoes are sweeter when sautéed and melt beautifully into eggs."
    },
    {
        "title": "Golden Cheesy Tomato Toast",
        "key_ingredients": ["bread", "cheese", "tomato"],
        "min_matches": 2,
        "prep_time_minutes": 8,
        "difficulty": "Easy",
        "steps": [
            "Toast bread lightly; slice tomatoes thinly.",
            "Layer tomatoes and cheese slices over the bread.",
            "Broil in oven or pan-fry covered on medium-low until cheese is bubbly and melted."
        ],
        "waste_reduction_tip": "Bread past its peak crispness revives wonderfully when toasted under melted cheese."
    },
    {
        "title": "Creamy Banana Protein Smoothie",
        "key_ingredients": ["banana", "milk"],
        "min_matches": 2,
        "prep_time_minutes": 5,
        "difficulty": "Easy",
        "steps": [
            "Peel bananas and slice into blender (can also freeze beforehand).",
            "Add milk (or yogurt) and optional pinch of cinnamon or honey.",
            "Blend on high for 45 seconds until silky smooth."
        ],
        "waste_reduction_tip": "Brown-spotted bananas contain higher natural sweetness, ideal for no-sugar smoothies."
    },
    {
        "title": "Quick Garden Vegetable Stir-Fry",
        "key_ingredients": ["carrot", "capsicum", "mushrooms", "onion", "spinach"],
        "min_matches": 2,
        "prep_time_minutes": 15,
        "difficulty": "Easy",
        "steps": [
            "Slice vegetables into thin, uniform strips.",
            "Heat 1 tbsp oil in a wok or large pan on high heat. Add onions and carrots first.",
            "Toss in capsicum, mushrooms, and greens with soy sauce or garlic; stir-fry for 4-5 mins."
        ],
        "waste_reduction_tip": "Stir-fries accommodate almost any vegetable combination before crispness is lost."
    },
    {
        "title": "Hearty Refrigerator Rescue Frittata",
        "key_ingredients": ["eggs", "spinach", "potato", "onion", "cheese"],
        "min_matches": 2,
        "prep_time_minutes": 20,
        "difficulty": "Medium",
        "steps": [
            "Parboil or pan-fry sliced potatoes and onions until tender.",
            "Fold in greens (spinach/herbs) until wilted.",
            "Pour beaten eggs over the mixture, top with cheese, and bake at 180°C (350°F) for 15 mins."
        ],
        "waste_reduction_tip": "Frittatas rescue multiple declining produce items simultaneously."
    },
    {
        "title": "Savory Garlic Butter Mushrooms on Toast",
        "key_ingredients": ["mushrooms", "bread", "butter"],
        "min_matches": 2,
        "prep_time_minutes": 10,
        "difficulty": "Easy",
        "steps": [
            "Slice mushrooms and melt butter in skillet over medium-high heat.",
            "Sear mushrooms without crowding until deep golden brown.",
            "Spoon over toasted bread and season with black pepper."
        ],
        "waste_reduction_tip": "Use mushrooms promptly once gills darken to enjoy rich umami flavor."
    },
    {
        "title": "Spiced Tomato & Egg Shakshuka",
        "key_ingredients": ["tomato", "eggs", "onion", "capsicum"],
        "min_matches": 2,
        "prep_time_minutes": 20,
        "difficulty": "Medium",
        "steps": [
            "Simmer chopped tomatoes, onions, and capsicum with cumin and paprika until saucy (10 mins).",
            "Make small wells in sauce and crack eggs directly into them.",
            "Cover and cook on low heat for 5-7 minutes until egg whites are set."
        ],
        "waste_reduction_tip": "Ideal for soft or bruised tomatoes; the sauce conceals any visual blemishes."
    },
    {
        "title": "Sweet Cinnamon Fried Apples",
        "key_ingredients": ["apple", "butter"],
        "min_matches": 2,
        "prep_time_minutes": 12,
        "difficulty": "Easy",
        "steps": [
            "Core and slice apples into wedges.",
            "Melt butter in a pan, add apple slices, cinnamon, and a spoonful of sugar.",
            "Cook over medium heat until apples are tender and caramelized (8-10 mins)."
        ],
        "waste_reduction_tip": "Apples that have lost crispness regain wonderful tenderness when warmed in butter."
    }
]

class RescueActionAgent:
    """
    Analyzes high-priority fridge items and formulates targeted rescue plans.
    """

    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("MODEL", "gemini-3.5-flash-lite")
        self.openai_key = os.getenv("OPENAI_API_KEY")

    def formulate_rescue_plan(self, ranked_inventory: Dict[str, Any]) -> Dict[str, Any]:
        items = ranked_inventory.get("items", [])
        rescue_now_items = [it for it in items if it.get("tier") == "RESCUE NOW"]
        use_soon_items = [it for it in items if it.get("tier") == "USE SOON"]

        target_items = rescue_now_items if rescue_now_items else (use_soon_items if use_soon_items else items[:3])
        if not target_items:
            return {
                "headline": "Fridge in Stable Condition",
                "summary": "All detected items have low spoilage risk. No emergency cooking required today.",
                "recommended_recipes": [],
                "engine_used": "Deterministic Logic"
            }

        # 1. Try Gemini GenAI Agent if key present
        if self.gemini_key and len(self.gemini_key.strip()) > 10:
            try:
                gemini_result = self._generate_with_gemini(target_items, items)
                if gemini_result:
                    return gemini_result
            except Exception as e:
                print(f"Notice: Gemini call failed ({e}), attempting fallback.")

        # 2. Try OpenAI if key present
        if self.openai_key and len(self.openai_key.strip()) > 10:
            try:
                llm_result = self._generate_with_openai(target_items, items)
                if llm_result:
                    return llm_result
            except Exception as e:
                print(f"Notice: OpenAI call failed ({e}), falling back to deterministic rescue engine.")

        # 3. Fallback heuristic engine
        return self._generate_with_heuristic(target_items, items)

    def _generate_with_gemini(self, priority_items: List[Dict[str, Any]], all_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        import requests
        import re

        clean_priority = [{"name": it["name"], "quantity": it["quantity"], "tier": it.get("tier"), "freshness": it.get("freshness")} for it in priority_items]
        clean_all = [{"name": it["name"], "quantity": it["quantity"]} for it in all_items]

        prompt = (
            "You are Fridge Rescue AI, an expert food-waste reduction culinary assistant under UN SDG 12. "
            "Given this refrigerator inventory, recommend a targeted rescue recipe that uses the HIGHEST URGENCY items first "
            "with realistic quantities and minimal additional pantry staples.\n\n"
            f"Highest Urgency Items: {json.dumps(clean_priority)}\n"
            f"All Available Items: {json.dumps(clean_all)}\n\n"
            "Return valid JSON ONLY (without markdown fences, or with standard json block) matching this schema:\n"
            "{\n"
            '  "headline": "Rescue Mission Title",\n'
            '  "summary": "1-2 sentence rationale prioritizing waste prevention",\n'
            '  "target_rescued_items": ["item1", "item2"],\n'
            '  "recommended_recipes": [{\n'
            '    "title": "Recipe Name",\n'
            '    "prep_time": "~12 minutes",\n'
            '    "difficulty": "Easy",\n'
            '    "ingredients_used": ["item1", "item2"],\n'
            '    "priority_rescued": ["item1"],\n'
            '    "instructions": ["Step 1", "Step 2", "Step 3"],\n'
            '    "waste_reduction_tip": "Why this prevents food waste"\n'
            "  }],\n"
            '  "impact_statement": "Prevents waste of N items"\n'
            "}"
        )

        model_name = self.gemini_model or "gemini-3.5-flash-lite"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }

        resp = requests.post(url, json=payload, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            # Extract JSON block if wrapped
            clean_json = re.sub(r'^```json\s*', '', content.strip(), flags=re.MULTILINE)
            clean_json = re.sub(r'```$', '', clean_json.strip())
            parsed = json.loads(clean_json)
            parsed["engine_used"] = f"Gemini GenAI ({model_name})"
            return parsed
        else:
            print(f"Gemini API returned status {resp.status_code}: {resp.text[:200]}")
            return None

    def _generate_with_openai(self, priority_items: List[Dict[str, Any]], all_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        import requests

        prompt = (
            "You are Fridge Rescue AI, an expert food waste prevention chef. "
            "Your PRIMARY goal is food waste reduction under UN SDG 12. "
            "Given the following fridge inventory, recommend an immediate recipe that uses the HIGHEST PRIORITY items first.\n\n"
            f"Highest Urgency Items: {json.dumps(priority_items)}\n"
            f"All Available Items: {json.dumps(all_items)}"
        )

        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a professional culinary food-waste prevention assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }

        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["engine_used"] = "OpenAI GenAI (GPT-4o-mini)"
            return parsed
        return None

    def _generate_with_heuristic(self, priority_items: List[Dict[str, Any]], all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        detected_names = [it["name"].lower() for it in all_items]
        priority_names = [it["name"].lower() for it in priority_items]

        # Score matching recipes from knowledge base
        matched_recipes = []
        for recipe in RESCUE_RECIPES_DB:
            keys = [k.lower() for k in recipe["key_ingredients"]]
            # Check overlap with priority items
            priority_matches = [k for k in keys if any(k in p for p in priority_names)]
            all_matches = [k for k in keys if any(k in a for a in detected_names)]

            score = len(priority_matches) * 2 + len(all_matches)
            if len(all_matches) >= recipe["min_matches"]:
                matched_recipes.append((score, priority_matches, all_matches, recipe))

        matched_recipes.sort(key=lambda x: x[0], reverse=True)

        if matched_recipes:
            best_score, pri_matches, all_matches, best_recipe = matched_recipes[0]
            headline = f"Rescue Mission: {best_recipe['title']}"
            summary = (
                f"Cook this first to rescue {', '.join(pri_matches if pri_matches else all_matches[:2])}. "
                f"Uses {len(all_matches)} ingredients currently in your fridge in ~{best_recipe['prep_time_minutes']} minutes."
            )
            recipe_obj = {
                "title": best_recipe["title"],
                "prep_time": f"~{best_recipe['prep_time_minutes']} minutes",
                "difficulty": best_recipe["difficulty"],
                "ingredients_used": all_matches,
                "priority_rescued": pri_matches,
                "instructions": best_recipe["steps"],
                "waste_reduction_tip": best_recipe["waste_reduction_tip"]
            }
            rec_list = [recipe_obj]
        else:
            # Generic fallback rescue recommendation
            primary_name = priority_items[0]["name"].capitalize()
            qty = priority_items[0]["quantity"]
            headline = f"Immediate Rescue: Prioritize {primary_name}"
            summary = (
                f"Your {primary_name} ({qty} units) shows the highest spoilage urgency. "
                "Consider roasting, sautéing, or freezing today to prevent spoilage."
            )
            rec_list = [{
                "title": f"Quick-Roasted / Sautéed {primary_name}",
                "prep_time": "~15 minutes",
                "difficulty": "Easy",
                "ingredients_used": [primary_name.lower()],
                "priority_rescued": [primary_name.lower()],
                "instructions": [
                    f"Slice {primary_name.lower()} into bite-sized pieces.",
                    "Toss with 1 tbsp cooking oil, salt, and seasonings of choice.",
                    "Sauté on medium-high heat until tender or bake at 200°C (400°F) for 15-20 mins."
                ],
                "waste_reduction_tip": f"Cooking preserves {primary_name.lower()} for an additional 3-4 days in airtight refrigeration."
            }]

        return {
            "headline": headline,
            "summary": summary,
            "target_rescued_items": [it["name"] for it in priority_items[:3]],
            "recommended_recipes": rec_list,
            "engine_used": "Deterministic Waste Prevention Engine (Offline Ready)",
            "impact_statement": f"Acting on this recommendation will prevent waste of {sum(it['quantity'] for it in priority_items[:3])} high-risk food item(s)."
        }

    def _generate_with_llm(self, priority_items: List[Dict[str, Any]], all_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        import requests
        
        prompt = (
            "You are Fridge Rescue AI, an expert food waste prevention chef. "
            "Your PRIMARY goal is food waste reduction under UN SDG 12. "
            "Given the following fridge inventory, recommend an immediate recipe that uses the HIGHEST PRIORITY items first "
            "with realistic quantities and minimal additional pantry staples.\n\n"
            f"Highest Urgency Items: {json.dumps(priority_items)}\n"
            f"All Available Items: {json.dumps(all_items)}\n\n"
            "Return valid JSON ONLY with format:\n"
            "{\n"
            '  "headline": "Brief rescue mission title",\n'
            '  "summary": "1-2 sentence rationale prioritizing waste prevention",\n'
            '  "target_rescued_items": ["item1", "item2"],\n'
            '  "recommended_recipes": [{\n'
            '    "title": "Recipe Name",\n'
            '    "prep_time": "~12 minutes",\n'
            '    "difficulty": "Easy",\n'
            '    "ingredients_used": ["tomato", "eggs"],\n'
            '    "priority_rescued": ["tomato"],\n'
            '    "instructions": ["Step 1", "Step 2", "Step 3"],\n'
            '    "waste_reduction_tip": "Why this prevents food waste"\n'
            "  }],\n"
            '  "impact_statement": "Preventing waste for N items"\n'
            "}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a professional culinary food-waste prevention assistant. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }

        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=12)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed["engine_used"] = "GenAI Agent (GPT-4o-mini)"
            return parsed
        return None
