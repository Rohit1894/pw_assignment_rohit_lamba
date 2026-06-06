# Task 03: Fix a Broken Workflow — Explanation

## What Was Actually Broken

The core problem was a **naming disconnect**: question and solution images inside ZIP files have random/hashed filenames (e.g., `a3f82b1c.png`, `IMG_20240115_142356.jpg`), while the mapping between these filenames and actual question numbers (Q1, Q2, etc.) lives in a separate Excel metadata file. Employees couldn't easily match the two, so they resorted to manually opening PDFs and taking screenshots — a tedious, error-prone process that defeats the purpose of having digital assets.

The workflow wasn't "broken" in the sense that something crashed — it was broken in that the **tooling gap forced manual workarounds**. The data existed in structured form (Excel + ZIP), but no script bridged the two.

## Approach

The script (`scripts/rename_questions.py`) works in three stages:

1. **Read the Excel metadata** — Auto-detect which columns contain the original filename and the question identifier. This uses pattern matching on column names (looking for variants like "filename", "file_name", "image", "question_no", "sr_no", etc.) so it works across different Excel layouts without hardcoding column indices.

2. **Extract images from the ZIP** — Filter out OS artifacts (`__MACOSX/`, `.DS_Store`), then match each image to its metadata row by comparing filenames (case-insensitive, with and without extensions).

3. **Rename to clean format** — Each file becomes `Q<n>.png` (question) or `S<n>.png` (solution). The Q/S classification uses multiple signals: filename keywords ("sol", "ans", "question"), a type column in the Excel if present, or sequential assignment as a fallback.

## Edge Cases Handled

- **Column name variations**: The auto-detection handles different naming conventions ("File Name", "filename", "Image", "Attachment", etc.) rather than requiring exact column names.
- **Missing question numbers**: If the Excel doesn't have explicit question numbers, the script assigns them sequentially based on row order.
- **Unknown Q/S type**: If neither the filename nor the metadata indicates whether a file is a question or solution, it defaults to "Question" and logs a warning.
- **Duplicate output names**: If two files would map to the same output name (e.g., two files both claim to be Q1), the second gets a `_2` suffix rather than overwriting.
- **macOS ZIP artifacts**: Files in `__MACOSX/` directories and dotfiles are filtered out automatically.
- **Mixed image formats**: Handles PNG, JPG, JPEG, GIF, BMP, and TIFF. Preserves the original extension.
- **Partial metadata**: Files in the ZIP that don't appear in the Excel are still processed using filename heuristics, and are flagged in the summary output.

## What Could Be Improved

- Support for nested folder structures inside the ZIP (currently assumes flat or simple nesting).
- A dry-run mode (`--dry-run`) that shows the mapping without actually copying files.
- GUI or web interface for non-technical employees.
- Direct PDF page extraction as a fallback if images aren't in the ZIP.
