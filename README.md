# Resume Analyzer & Job Matcher

An NLP project that compares a resume (PDF) with a job description and returns a match score,
matched/missing skills, candidate details, and LLM-written improvement tips.

## Features
- PDF text extraction (pdfplumber)
- Text cleaning: lowercase, stopwords, lemmatization (spaCy)
- Skill extraction from an editable `skills.txt`
- Keyword similarity (TF-IDF + cosine) and semantic similarity (Sentence-Transformers)
- Named Entity Recognition for candidate name, plus regex for email, phone, degree
- LLM feedback via the Anthropic API (falls back to rule-based tips with no key)
- Streamlit UI with metrics, chart and keyword highlighting

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
export ANTHROPIC_API_KEY=your_key   # optional
streamlit run app.py
```

## How the score works
Overall = 50% skill coverage + 30% embedding similarity + 20% TF-IDF similarity.

## Project structure
```
app.py                 Streamlit UI
utils/pdf_reader.py    PDF to text
utils/preprocess.py    cleaning, lemmatization
utils/entities.py      name/email/phone/degree extraction
utils/matcher.py       skills, TF-IDF, embeddings, final score
utils/llm_feedback.py  LLM suggestions
skills.txt             skill dictionary (add your own)
```

## Future work
- Store many job descriptions in a vector DB (Chroma/FAISS) and recommend the best-fit job (RAG)
- Support DOCX resumes and OCR for scanned PDFs
- Section-wise scoring (education, experience, projects)
