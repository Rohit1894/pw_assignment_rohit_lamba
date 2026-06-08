import os
import sys

# Let scripts be imported as modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from ocr_question import extract_question_info
from render_video import render_video

# Paths for new question
image_path = "input/new_question.png"
audio_path = "input/new_narration.mp3"
annotations_path = "output/new_annotations.json"
output_path = "output/new_final.mp4"

print("1. Running OCR on new question image...")
question_text, option_positions, question_bbox, enriched_ocr = extract_question_info(image_path)

print("2. Rendering new final video...")
render_video(
    image_path, annotations_path, audio_path,
    output_path, option_positions, question_bbox, enriched_ocr,
)

print("\nDone! New video saved to:", os.path.abspath(output_path))
