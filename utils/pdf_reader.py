import pdfplumber


def extract_text_from_pdf(file) -> str:
    """Read all pages of a PDF (path or file-like object) and return the text."""
    pages = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n".join(pages).strip()
