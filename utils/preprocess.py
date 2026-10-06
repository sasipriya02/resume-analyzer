import re
import spacy

_nlp = None


def get_nlp():
    """Load spaCy once. Falls back to a blank pipeline if the model is missing."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            _nlp = spacy.blank("en")
    return _nlp


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/emails/symbols, drop stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    doc = get_nlp()(text)
    tokens = []
    for tok in doc:
        if tok.is_space or tok.is_stop or len(tok.text) < 2:
            continue
        lemma = tok.lemma_.strip() if tok.lemma_ else tok.text
        tokens.append(lemma if lemma else tok.text)
    return " ".join(tokens)
