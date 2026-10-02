"""
Module: ml/ocr/expiry_detector.py
Purpose: Extract and normalize expiry dates from packaged food items using EasyOCR
"""

import re
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2
import numpy as np

class ExpiryDateDetector:
    """
    Detects packaging text and extracts expiry dates (Best Before, Use By, EXP, BB, MFG).
    Normalizes recognized dates into standard ISO YYYY-MM-DD format.
    """

    EXPIRY_KEYWORDS = [
        "best before", "best by", "use by", "exp", "expiry", 
        "exp date", "bb", "bb date", "mfg", "mfg date", "consume by"
    ]

    # Date regex patterns supporting multiple international packaging conventions
    DATE_PATTERNS = [
        # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
        (r'\b(20[2-3][0-9])[-/.](0[1-9]|1[0-2])[-/.](0[1-9]|[12][0-9]|3[01])\b', '%Y-%m-%d'),
        # DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
        (r'\b(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[0-2])[-/.](20[2-3][0-9])\b', '%d-%m-%Y'),
        # DD-MM-YY or DD/MM/YY or DD.MM.YY
        (r'\b(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[0-2])[-/.]([2-3][0-9])\b', '%d-%m-%y'),
        # Month name abbreviations (e.g., 08 OCT 2026, 15-NOV-25, OCT 2026)
        (r'\b(0[1-9]|[12][0-9]|3[01])?\s*[-/.]?\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*\s*[-/.]?\s*(20[2-3][0-9]|[2-3][0-9])\b', 'alpha'),
        # MM/YY (e.g., 10/26, 12/27)
        (r'\b(0[1-9]|1[0-2])[-/.]([2-3][0-9])\b', '%m-%y')
    ]

    MONTH_MAP = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        self._reader = None
        self.languages = languages
        self.gpu = gpu

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            # EasyOCR lazy initialization
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def parse_alpha_date(self, match_tuple: tuple) -> Optional[str]:
        day, month_str, year = match_tuple
        month = self.MONTH_MAP.get(month_str.lower()[:3])
        if not month:
            return None
        
        day_num = int(day) if day else 1
        year_num = int(year)
        if year_num < 100:
            year_num += 2000

        try:
            d = datetime.date(year_num, month, day_num)
            return d.strftime("%Y-%m-%d")
        except ValueError:
            return None

    def normalize_date_string(self, text: str) -> Optional[Dict[str, Any]]:
        clean_text = text.upper()
        
        # Test regex patterns
        for pattern, fmt in self.DATE_PATTERNS:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                matched_raw = match.group(0)
                try:
                    if fmt == 'alpha':
                        iso = self.parse_alpha_date(match.groups())
                        if iso:
                            return {"raw": matched_raw, "iso": iso}
                    elif fmt == '%d-%m-%y':
                        # normalize separators
                        norm_str = re.sub(r'[/.]', '-', matched_raw)
                        d = datetime.datetime.strptime(norm_str, '%d-%m-%y').date()
                        return {"raw": matched_raw, "iso": d.strftime("%Y-%m-%d")}
                    elif fmt == '%d-%m-%Y':
                        norm_str = re.sub(r'[/.]', '-', matched_raw)
                        d = datetime.datetime.strptime(norm_str, '%d-%m-%Y').date()
                        return {"raw": matched_raw, "iso": d.strftime("%Y-%m-%d")}
                    elif fmt == '%Y-%m-%d':
                        norm_str = re.sub(r'[/.]', '-', matched_raw)
                        d = datetime.datetime.strptime(norm_str, '%Y-%m-%d').date()
                        return {"raw": matched_raw, "iso": d.strftime("%Y-%m-%d")}
                    elif fmt == '%m-%y':
                        norm_str = re.sub(r'[/.]', '-', matched_raw)
                        d = datetime.datetime.strptime(norm_str, '%m-%y').date()
                        # default to last day or middle of month
                        d = d.replace(day=28)
                        return {"raw": matched_raw, "iso": d.strftime("%Y-%m-%d")}
                except Exception:
                    continue
        return None

    def extract_expiry(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Processes image/crop and looks for expiry date markers and normalized date.
        """
        try:
            results = self.reader.readtext(image_np)
        except Exception as e:
            return {
                "detected": False,
                "raw_text": "",
                "expiry_date": None,
                "confidence": 0.0,
                "keyword_found": None,
                "error": str(e)
            }

        all_text = []
        found_keyword = None
        detected_date = None
        best_conf = 0.0

        for (bbox, text, conf) in results:
            all_text.append(text)
            text_lower = text.lower()
            
            # Check for keyword
            for kw in self.EXPIRY_KEYWORDS:
                if kw in text_lower:
                    found_keyword = kw
                    break

            # Try date match directly on chunk
            date_match = self.normalize_date_string(text)
            if date_match and conf > best_conf:
                detected_date = date_match["iso"]
                best_conf = float(conf)

        # If not found in individual chunks, join text and check full string
        full_text = " ".join(all_text)
        if not detected_date:
            date_match = self.normalize_date_string(full_text)
            if date_match:
                detected_date = date_match["iso"]
                best_conf = 0.65

        # Calculate days remaining if date was found
        days_remaining = None
        if detected_date:
            try:
                target_dt = datetime.datetime.strptime(detected_date, "%Y-%m-%d").date()
                today = datetime.date.today()
                days_remaining = (target_dt - today).days
            except Exception:
                pass

        return {
            "detected": detected_date is not None,
            "raw_text": full_text[:200],
            "expiry_date": detected_date,
            "days_remaining": days_remaining,
            "confidence": round(best_conf, 2),
            "keyword_found": found_keyword
        }
