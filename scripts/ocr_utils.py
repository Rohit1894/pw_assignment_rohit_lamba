#!/usr/bin/env python3
"""
OCR Utilities for Physics Wallah Teacher Simulation.
Includes:
1. Structured OCRElement class with type classification.
2. OCRIndex with fuzzy keyword/text searching.
3. Geometry algorithm for finding the Largest Empty Rectangle.
"""

import numpy as np
import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Any


class OCRElement:
    """Represents a single OCR-detected text element with semantic classification."""
    
    def __init__(self, bbox: List[List[float]], text: str, confidence: float, index: int):
        self.bbox = [[int(coord) for coord in pt] for pt in bbox]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        self.text = text.strip()
        self.confidence = confidence
        self.index = index
        
        # Extents
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        self.x1, self.y1 = int(min(xs)), int(min(ys))
        self.x2, self.y2 = int(max(xs)), int(max(ys))
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        
        # Classification
        self.type = self._classify_type()

    def _classify_type(self) -> str:
        text_lower = self.text.lower()
        
        # Option pattern e.g., (A) 3 units, (a), B.
        if re.search(r'^\s*[\(\[-]?([a-dA-D])[\)\]\.-]?', self.text) or text_lower in ["(a)", "(b)", "(c)", "(d)"]:
            return "option"
            
        # Coordinate pattern e.g., (1, 2) or (4, 6) or A (1, 2)
        if re.search(r'\(?\d+\s*,\s*\d+\)?', text_lower) or re.search(r'\b[A-Za-z]\s*\(?\d+', text_lower):
            return "coordinate"
            
        # Formula reference e.g., "distance formula"
        if "formula" in text_lower or "theorem" in text_lower:
            return "formula_reference"
            
        # Diagram heuristic (low confidence, short letters, or outlying coords)
        if self.confidence < 0.25 and len(self.text) <= 4:
            return "diagram"
            
        return "text"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary matching the target format."""
        return {
            "index": self.index,
            "text": self.text,
            "bbox": self.bbox,
            "type": self.type,
            "confidence": self.confidence,
            "bounds": [self.x1, self.y1, self.x2, self.y2]
        }


class OCRIndex:
    """Fuzzy lookup index for finding OCR elements by text query."""
    
    def __init__(self, elements: List[OCRElement]):
        self.elements = elements
        
    def find_by_text(self, query: str, threshold: float = 0.5) -> List[OCRElement]:
        """Find OCR elements matching a query with fuzzy fallback."""
        query_lower = query.lower().strip()
        matches = []
        
        for elem in self.elements:
            elem_text = elem.text.lower().strip()
            
            # Exact match
            if elem_text == query_lower:
                matches.append((elem, 1.0))
                continue
                
            # Substring match
            if query_lower in elem_text or elem_text in query_lower:
                matches.append((elem, 0.9))
                continue
                
            # Fuzzy match
            ratio = SequenceMatcher(None, query_lower, elem_text).ratio()
            if ratio >= threshold:
                matches.append((elem, ratio))
                
        # Sort matches by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]


def find_largest_empty_rectangle(width: int, height: int, elements: List[OCRElement], question_bbox: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
    """
    Geometry algorithm to find maximal empty rectangles.
    Treats all OCR bounding boxes as occupied.
    Prioritizes and outputs empty regions.
    """
    obstacles = []
    diagram_bbox = None
    
    # Extract occupied boxes
    for elem in elements:
        # Pad obstacles slightly
        pad = 8
        ox1 = max(0, elem.x1 - pad)
        oy1 = max(0, elem.y1 - pad)
        ox2 = min(width, elem.x2 + pad)
        oy2 = min(height, elem.y2 + pad)
        obstacles.append((ox1, oy1, ox2, oy2))
        
        if elem.type == "diagram" or (elem.confidence < 0.25 and elem.text.lower() == "jd"):
            diagram_bbox = (elem.x1, elem.y1, elem.x2, elem.y2)

    # Vertical coordinates candidate boundaries
    left_coords = [0] + [x2 for (x1, y1, x2, y2) in obstacles]
    right_coords = [width] + [x1 for (x1, y1, x2, y2) in obstacles]
    
    candidates = []
    
    for xa in left_coords:
        if xa < 0 or xa >= width:
            continue
        for xb in right_coords:
            if xb <= xa or xb > width:
                continue
                
            # Find obstacles that overlap with the open vertical strip (xa, xb)
            strip_obstacles = [o for o in obstacles if o[0] < xb and o[2] > xa]
            
            # Sort y intervals of strip obstacles
            y_intervals = sorted([(o[1], o[3]) for o in strip_obstacles], key=lambda item: item[0])
            
            # Find empty gaps in y direction
            curr_y = 0
            for (oy1, oy2) in y_intervals:
                if oy1 > curr_y:
                    candidates.append((xa, curr_y, xb, oy1))
                curr_y = max(curr_y, oy2)
            if height > curr_y:
                candidates.append((xa, curr_y, xb, height))

    # Evaluate candidates
    categorized_regions = []
    seen = set()
    
    # Fallback to question bbox if none
    qx1, qy1, qx2, qy2 = question_bbox if question_bbox else (0, 0, width // 2, height // 3)
    
    # Heuristic diagram bbox if not found
    if not diagram_bbox:
        # Look for any outlying boxes or use center-right
        diagram_bbox = (width // 2 - 100, height // 2 - 100, width // 2 + 100, height // 2 + 100)

    for (x1, y1, x2, y2) in candidates:
        w_rect = x2 - x1
        h_rect = y2 - y1
        area = w_rect * h_rect
        
        # Filter out thin slivers
        if w_rect < 180 or h_rect < 120:
            continue
            
        rect_tuple = (x1, y1, x2, y2)
        if rect_tuple in seen:
            continue
        seen.add(rect_tuple)
        
        # Categorize
        # Priority 1: Right side of question (x1 is past question x2 or in right 40% of screen, and y is not too low)
        if x1 >= qx2 - 50 or (x1 >= width * 0.45 and y1 <= qy2 + 250):
            pos = "right"
            priority = 1
        # Priority 2: Below diagram (diagram bottom is at dy2)
        elif y1 >= diagram_bbox[3] - 20:
            pos = "bottom_diagram"
            priority = 2
        # Priority 3: Below question (y1 is past question y2)
        elif y1 >= qy2 - 10:
            pos = "bottom_question"
            priority = 3
        # Priority 4: Secondary
        else:
            pos = "secondary"
            priority = 4
            
        categorized_regions.append({
            "bounds": [x1, y1, x2, y2],
            "area": area,
            "width": w_rect,
            "height": h_rect,
            "position": pos,
            "priority": priority
        })

    # Sort by priority ascending (1 = highest), then by area descending
    categorized_regions.sort(key=lambda r: (r["priority"], -r["area"]))
    return categorized_regions


def enrich_ocr_data(ocr_results: List[Tuple], image_width: int, image_height: int, question_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """Enrich raw EasyOCR output with types, search index, and largest empty rectangle detection."""
    elements = []
    for idx, (bbox, text, conf) in enumerate(ocr_results):
        elements.append(OCRElement(bbox, text, conf, idx))
        
    ocr_index = OCRIndex(elements)
    free_spaces = find_largest_empty_rectangle(image_width, image_height, elements, question_bbox)
    
    return {
        "elements": [elem.to_dict() for elem in elements],
        "index": ocr_index,
        "free_spaces": free_spaces,
        "full_text": " ".join([elem.text for elem in elements]),
        "element_count": len(elements)
    }
