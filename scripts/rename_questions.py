#!/usr/bin/env python3
"""
Task 03: Fix a Broken Workflow — Question File Renamer

Problem:
    Employees manually screenshot questions from PDFs because ZIP files
    contain randomly-named image files and the Excel metadata mapping
    filenames to question numbers is hard to parse by hand.

Solution:
    This script reads the Excel metadata, extracts images from the ZIP,
    and renames them to a clean Q1.png, S1.png, Q2.png, S2.png, ... format.

Usage:
    python scripts/rename_questions.py
    python scripts/rename_questions.py --zip input/questions.zip --excel input/metadata.xlsx
    python scripts/rename_questions.py --zip input/questions.zip --excel input/metadata.xlsx --output output/renamed/
"""

import argparse
import os
import re
import shutil
import sys
import zipfile

import pandas as pd


def find_mapping_columns(df):
    """
    Auto-detect which columns contain the filename and question identifier.

    Looks for columns whose names suggest they hold filenames (e.g.,
    'filename', 'file_name', 'image') and question IDs ('question',
    'q_no', 'number', 'id').

    Returns:
        tuple: (filename_col, question_id_col) or (None, None) if not found.
    """
    filename_patterns = [
        r"file\s*name", r"image\s*name", r"image", r"file",
        r"source", r"path", r"attachment",
    ]
    question_patterns = [
        r"question\s*(no|num|number|id)?", r"q\s*\.?\s*(no|num|number|id)?",
        r"sr\s*\.?\s*(no|num)?", r"serial", r"number", r"id", r"sl\s*\.?\s*no",
    ]

    cols = list(df.columns)
    cols_lower = [str(c).lower().strip() for c in cols]

    filename_col = None
    question_col = None

    for pat in filename_patterns:
        for i, c in enumerate(cols_lower):
            if re.search(pat, c):
                filename_col = cols[i]
                break
        if filename_col:
            break

    for pat in question_patterns:
        for i, c in enumerate(cols_lower):
            if re.search(pat, c):
                question_col = cols[i]
                break
        if question_col:
            break

    return filename_col, question_col


def detect_type(filename, row_data=None):
    """
    Determine if a file is a Question or Solution image.

    Heuristics:
      - Filename contains 'sol', 'ans', 'solution', 'answer' -> Solution
      - Filename contains 'q', 'que', 'question' -> Question
      - If row_data has a 'type' column, use that
      - Default: alternate (odd = Question, even = Solution)
    """
    name_lower = filename.lower()

    if any(kw in name_lower for kw in ["sol", "ans", "solution", "answer"]):
        return "S"
    if any(kw in name_lower for kw in ["que", "question", "ques"]):
        return "Q"

    # Check if row data has a type indicator
    if row_data is not None:
        for col in row_data.index:
            val = str(row_data[col]).lower().strip()
            if val in ("question", "q", "que"):
                return "Q"
            if val in ("solution", "s", "sol", "answer", "ans"):
                return "S"

    return None  # Unknown — caller will decide


def extract_question_number(value):
    """
    Extract a numeric question number from a cell value.

    Handles formats like: '1', 'Q1', 'Q.1', 'Question 1', '01', etc.
    """
    s = str(value).strip()
    match = re.search(r"(\d+)", s)
    if match:
        return int(match.group(1))
    return None


def rename_questions(zip_path, excel_path, output_dir):
    """
    Main function: read metadata, extract ZIP, rename files.

    Args:
        zip_path:   Path to the ZIP containing question/solution images.
        excel_path: Path to the Excel file with filename-to-question mapping.
        output_dir: Directory to write renamed files into.

    Returns:
        list: Summary of (original_name, new_name) pairs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Read Excel metadata ──────────────────────────────────────────────
    print(f"  Reading metadata from {excel_path}...")

    # Try all sheets; use the first one that has meaningful data
    xls = pd.ExcelFile(excel_path)
    df = None
    for sheet in xls.sheet_names:
        candidate = pd.read_excel(xls, sheet_name=sheet)
        if len(candidate) > 0 and len(candidate.columns) >= 2:
            df = candidate
            print(f"  Using sheet: '{sheet}' ({len(df)} rows, {len(df.columns)} cols)")
            break

    if df is None:
        print("  ERROR: No usable sheet found in Excel file.")
        return []

    # ── Detect columns ───────────────────────────────────────────────────
    filename_col, question_col = find_mapping_columns(df)

    if not filename_col:
        print(f"  WARNING: Could not auto-detect filename column.")
        print(f"  Available columns: {list(df.columns)}")
        # Fall back to the first column
        filename_col = df.columns[0]
        print(f"  Falling back to first column: '{filename_col}'")

    if not question_col:
        print(f"  WARNING: Could not auto-detect question ID column.")
        # Fall back to the second column or row index
        if len(df.columns) >= 2:
            question_col = df.columns[1]
            print(f"  Falling back to second column: '{question_col}'")

    print(f"  Filename column: '{filename_col}'")
    print(f"  Question ID column: '{question_col}'")

    # ── Build the mapping ────────────────────────────────────────────────
    file_to_info = {}  # original_filename -> (question_number, type)

    for idx, row in df.iterrows():
        original_name = str(row[filename_col]).strip()
        if not original_name or original_name.lower() == "nan":
            continue

        q_num = None
        if question_col:
            q_num = extract_question_number(row[question_col])

        file_type = detect_type(original_name, row)
        file_to_info[original_name] = (q_num, file_type)

    print(f"  Found {len(file_to_info)} file mappings in metadata")

    # ── Extract ZIP ──────────────────────────────────────────────────────
    print(f"  Extracting images from {zip_path}...")

    if not zipfile.is_zipfile(zip_path):
        print(f"  ERROR: {zip_path} is not a valid ZIP file.")
        return []

    results = []
    unmatched = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        image_files = [
            f for f in zf.namelist()
            if not f.startswith("__MACOSX")
            and not f.startswith(".")
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"))
        ]

        print(f"  Found {len(image_files)} images in ZIP")

        # Match ZIP entries to metadata
        q_counter = 1  # For files without explicit question numbers
        s_counter = 1

        for img_path in sorted(image_files):
            img_name = os.path.basename(img_path)
            ext = os.path.splitext(img_name)[1].lower()
            # Normalize extension to .png for consistency
            out_ext = ext if ext else ".png"

            # Try to find this file in the metadata
            matched = False
            for meta_name, (q_num, ftype) in file_to_info.items():
                # Match by exact name or basename
                if (img_name == meta_name
                        or img_name.lower() == meta_name.lower()
                        or os.path.splitext(img_name)[0].lower()
                        == os.path.splitext(meta_name)[0].lower()):

                    if ftype is None:
                        ftype = "Q"  # Default to Question if unknown
                    if q_num is None:
                        # Assign sequential number
                        if ftype == "Q":
                            q_num = q_counter
                            q_counter += 1
                        else:
                            q_num = s_counter
                            s_counter += 1

                    new_name = f"{ftype}{q_num}{out_ext}"
                    matched = True
                    break

            if not matched:
                # No metadata match — use filename heuristics
                ftype = detect_type(img_name)
                q_num_from_name = extract_question_number(img_name)

                if ftype and q_num_from_name:
                    new_name = f"{ftype}{q_num_from_name}{out_ext}"
                elif ftype == "S":
                    new_name = f"S{s_counter}{out_ext}"
                    s_counter += 1
                else:
                    new_name = f"Q{q_counter}{out_ext}"
                    q_counter += 1
                unmatched.append(img_name)

            # Extract and rename
            out_path = os.path.join(output_dir, new_name)

            # Handle duplicates by appending a suffix
            if os.path.exists(out_path):
                base, ext_part = os.path.splitext(new_name)
                i = 2
                while os.path.exists(os.path.join(output_dir, f"{base}_{i}{ext_part}")):
                    i += 1
                new_name = f"{base}_{i}{ext_part}"
                out_path = os.path.join(output_dir, new_name)

            # Write the file
            with zf.open(img_path) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            results.append((img_name, new_name))

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  Renaming complete!")
    print(f"  Total files processed: {len(results)}")
    print(f"  Matched via metadata:  {len(results) - len(unmatched)}")
    print(f"  Matched via heuristics: {len(unmatched)}")

    print(f"\n  Output directory: {os.path.abspath(output_dir)}")
    print(f"  ─────────────────────────────────────")
    for orig, new in sorted(results, key=lambda x: x[1]):
        print(f"    {orig:40s} -> {new}")

    if unmatched:
        print(f"\n  Files not found in metadata (used filename heuristics):")
        for name in unmatched:
            print(f"    - {name}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Rename randomly-named question/solution images using Excel metadata"
    )
    parser.add_argument(
        "--zip", default="input/questions.zip",
        help="Path to the ZIP file containing images"
    )
    parser.add_argument(
        "--excel", default="input/metadata.xlsx",
        help="Path to the Excel metadata file"
    )
    parser.add_argument(
        "--output", default="output/renamed",
        help="Output directory for renamed files"
    )
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        print(f"ERROR: ZIP file not found: {args.zip}")
        print("Please provide the path with --zip")
        sys.exit(1)

    if not os.path.exists(args.excel):
        print(f"ERROR: Excel file not found: {args.excel}")
        print("Please provide the path with --excel")
        sys.exit(1)

    rename_questions(args.zip, args.excel, args.output)


if __name__ == "__main__":
    main()
