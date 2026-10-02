"""
Test Suite: tests/test_ocr.py
Purpose: Test regex date parsing and normalization logic
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ocr.expiry_detector import ExpiryDateDetector

def test_date_normalization():
    detector = ExpiryDateDetector()

    # Test cases: raw packaging string -> expected ISO YYYY-MM-DD
    cases = [
        ("BEST BEFORE 2026-10-08", "2026-10-08"),
        ("EXP: 15/05/2027", "2027-05-15"),
        ("USE BY 12-04-26", "2026-04-12"),
        ("BB 08 OCT 2026", "2026-10-08"),
        ("EXP 11/26", "2026-11-28")
    ]

    for raw, expected in cases:
        norm = detector.normalize_date_string(raw)
        assert norm is not None, f"Failed to match: {raw}"
        assert norm["iso"] == expected, f"Expected {expected}, got {norm['iso']} for {raw}"
        print(f"PASS: '{raw}' -> '{norm['iso']}'")

if __name__ == "__main__":
    test_date_normalization()
    print("All OCR date normalization tests passed!")
