import re
from .preprocess import get_nlp

DEGREE_PATTERN = re.compile(
    r"\b(b\.?\s?tech|b\.?\s?e\.?|b\.?\s?sc|b\.?\s?c\.?a|m\.?\s?tech|m\.?\s?e\.?|m\.?\s?sc|m\.?\s?c\.?a|"
    r"mba|bachelor[^\n,.]{0,40}|master[^\n,.]{0,40})\b",
    re.IGNORECASE,
)


def extract_entities(text: str) -> dict:
    """Pull basic details out of a resume: name, email, phone, degrees."""
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone = re.search(r"(?:\+91[\s-]?)?[6-9]\d{9}", text.replace(" ", ""))

    name = None
    doc = get_nlp()(text[:400])  # name is almost always at the top
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            break
    if not name:  # fallback: first non-empty line
        first = next((l.strip() for l in text.splitlines() if l.strip()), "")
        name = first if len(first.split()) <= 4 else None

    degrees = sorted({m.group(0).strip() for m in DEGREE_PATTERN.finditer(text)})
    return {
        "name": name,
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "degrees": degrees,
    }
