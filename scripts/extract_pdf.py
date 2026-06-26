import sys
import os

def try_extract():
    pdf_path = "e:/ai automation/pw_task_1/pw_assignment_rohit_lamba/input/ppt/PPT_मानव जनन (Human Reproduction)_Zoology.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        return
        
    print("Testing libraries...")
    
    # Try pypdf
    try:
        import pypdf
        print("Using pypdf...")
        reader = pypdf.PdfReader(pdf_path)
        print(f"Total pages: {len(reader.pages)}")
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"--- Page {idx+1} ---")
            print(text[:500])
        return
    except ImportError:
        print("pypdf not installed")
        
    # Try fitz (PyMuPDF)
    try:
        import fitz
        print("Using PyMuPDF (fitz)...")
        doc = fitz.open(pdf_path)
        print(f"Total pages: {len(doc)}")
        for idx, page in enumerate(doc):
            text = page.get_text()
            print(f"--- Page {idx+1} ---")
            print(text[:500])
        return
    except ImportError:
        print("fitz not installed")

    # Try pdfplumber
    try:
        import pdfplumber
        print("Using pdfplumber...")
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Total pages: {len(pdf.pages)}")
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text()
                print(f"--- Page {idx+1} ---")
                print(text[:500])
        return
    except ImportError:
        print("pdfplumber not installed")
        
    print("No PDF extraction libraries installed. Installing pypdf...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pypdf"])
    
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        print(f"Total pages: {len(reader.pages)}")
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            print(f"--- Page {idx+1} ---")
            print(text[:500])
    except Exception as e:
        print(f"Error extracting after installation: {e}")

if __name__ == "__main__":
    try_extract()
