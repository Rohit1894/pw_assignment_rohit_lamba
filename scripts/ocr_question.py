#!/usr/bin/env python3
"""Extract question text and option positions from question image using EasyOCR."""

import easyocr
import json
import sys


def extract_question_info(image_path):
    """
    Extract text content, option bounding boxes, and question region from a
    question image.

    Returns:
        tuple: (full_text, option_positions, question_bbox)
            - full_text: All OCR text concatenated
            - option_positions: Dict mapping option letters to bounding boxes
              e.g. {"A": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ...}
            - question_bbox: (x1, y1, x2, y2) bounding box covering the
              question text region (everything above the options), or None
    """
    reader = easyocr.Reader(["en"], verbose=False)
    results = reader.readtext(image_path)

    full_text = " ".join([text for _, text, _ in results])

    option_positions = {}
    option_y_positions = []

    for bbox, text, confidence in results:
        text_lower = text.lower().strip()
        for opt in ["a", "b", "c", "d"]:
            if f"({opt})" in text_lower:
                option_positions[opt.upper()] = bbox
                # Track the top-y of option regions
                option_y_positions.append(min(p[1] for p in bbox))
                break

    # Compute the question text bounding box: the region above the options
    question_bbox = None
    if results:
        all_x = []
        all_y = []
        for bbox, _, _ in results:
            for p in bbox:
                all_x.append(p[0])
                all_y.append(p[1])

        if option_y_positions:
            # Question region = everything above the first option
            options_top = min(option_y_positions)
            q_y_max = options_top - 5
        else:
            # No options detected — treat the entire text area as question
            q_y_max = max(all_y)

        question_bbox = (
            int(min(all_x)),
            int(min(all_y)),
            int(max(all_x)),
            int(q_y_max),
        )

    print(f"  OCR extracted {len(results)} text regions")
    if option_positions:
        print(f"  Found options: {list(option_positions.keys())}")
    if question_bbox:
        print(f"  Question region: {question_bbox}")

    return full_text, option_positions, question_bbox


if __name__ == "__main__":
    image = sys.argv[1] if len(sys.argv) > 1 else "input/question.png"
    text, positions, q_bbox = extract_question_info(image)
    print(f"\nQuestion text:\n{text}")
    print(f"\nOption positions:\n{json.dumps(positions, indent=2)}")
    print(f"\nQuestion bounding box:\n{q_bbox}")
