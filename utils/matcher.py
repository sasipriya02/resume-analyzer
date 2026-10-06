import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import clean_text

SKILLS_FILE = Path(__file__).resolve().parent.parent / "skills.txt"
_model = None


def load_skills() -> list[str]:
    return [s.strip().lower() for s in SKILLS_FILE.read_text().splitlines() if s.strip()]


def extract_skills(text: str, skills: list[str] | None = None) -> set[str]:
    """Find known skills in text. Boundaries are chosen so 'c++' and 'c#' work."""
    skills = skills or load_skills()
    text = text.lower()
    found = set()
    for skill in skills:
        pattern = r"(?<![\w+#])" + re.escape(skill) + r"(?![\w+#])"
        if re.search(pattern, text):
            found.add(skill)
    return found


def tfidf_score(resume: str, jd: str) -> float:
    """Keyword-based similarity (0-100)."""
    vec = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vec.fit_transform([clean_text(resume), clean_text(jd)])
    return round(float(cosine_similarity(tfidf[0], tfidf[1])[0][0]) * 100, 2)


def embedding_score(resume: str, jd: str) -> float:
    """Meaning-based similarity (0-100) using Sentence-Transformers."""
    global _model
    from sentence_transformers import SentenceTransformer

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = _model.encode([resume, jd], normalize_embeddings=True)
    score = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    return round(max(score, 0.0) * 100, 2)


def analyze(resume: str, jd: str, use_embeddings: bool = True) -> dict:
    skills = load_skills()
    resume_skills = extract_skills(resume, skills)
    jd_skills = extract_skills(jd, skills)

    matched = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    coverage = round(len(matched) / len(jd_skills) * 100, 2) if jd_skills else 0.0

    tfidf = tfidf_score(resume, jd)
    emb = embedding_score(resume, jd) if use_embeddings else None

    parts = [(coverage, 0.5), (tfidf, 0.2)]
    if emb is not None:
        parts.append((emb, 0.3))
    total_w = sum(w for _, w in parts)
    final = round(sum(s * w for s, w in parts) / total_w, 2)

    return {
        "final_score": final,
        "skill_coverage": coverage,
        "tfidf_score": tfidf,
        "embedding_score": emb,
        "matched": matched,
        "missing": missing,
        "extra": extra,
    }
