from PyPDF2 import PdfReader
from pathlib import Path


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def load_papers_from_dir(pdf_dir: str):
    pdf_dir = Path(pdf_dir)
    pdf_paths = [str(f) for f in pdf_dir.iterdir() if f.suffix.lower() == ".pdf"]
    return pdf_paths