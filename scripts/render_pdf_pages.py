import sys
import os
import subprocess

def render_pdf_pages():
    pdf_path = "e:/ai automation/pw_task_1/pw_assignment_rohit_lamba/input/ppt/PPT_मानव जनन (Human Reproduction)_Zoology.pdf"
    output_dir = "e:/ai automation/pw_task_1/pw_assignment_rohit_lamba/output/analysis/pdf_pages"
    os.makedirs(output_dir, exist_ok=True)
    
    # Install pymupdf (fitz) if not present
    try:
        import fitz
    except ImportError:
        print("Installing PyMuPDF...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pymupdf"])
        import fitz
        
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    
    # Save the first 15 pages as images to inspect
    for i in range(min(15, len(doc))):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        out_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    render_pdf_pages()
