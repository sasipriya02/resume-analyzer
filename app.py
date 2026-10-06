import re
import pandas as pd
import streamlit as st

from utils.pdf_reader import extract_text_from_pdf
from utils.entities import extract_entities
from utils.matcher import analyze
from utils.llm_feedback import get_feedback

st.set_page_config(page_title="Resume Analyzer & Job Matcher", page_icon="📄", layout="wide")
st.title("📄 Resume Analyzer & Job Matcher")
st.caption("NLP project: TF-IDF + embeddings + skill extraction + LLM feedback")

with st.sidebar:
    st.header("Settings")
    use_emb = st.checkbox("Use embeddings (downloads a ~90MB model on first run)", value=True)
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    st.markdown("Without a key you still get the scores and a basic rule-based feedback.")

col1, col2 = st.columns(2)
with col1:
    pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"])
with col2:
    jd = st.text_area("Paste job description", height=220)

if st.button("Analyze", type="primary"):
    if not pdf or not jd.strip():
        st.warning("Please upload a resume and paste a job description.")
        st.stop()

    with st.spinner("Reading resume..."):
        resume = extract_text_from_pdf(pdf)
    if not resume:
        st.error("Could not read text from this PDF. It may be a scanned image.")
        st.stop()

    with st.spinner("Analyzing..."):
        result = analyze(resume, jd, use_embeddings=use_emb)
        info = extract_entities(resume)

    st.subheader("Match score")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall", f"{result['final_score']}%")
    c2.metric("Skill coverage", f"{result['skill_coverage']}%")
    c3.metric("TF-IDF", f"{result['tfidf_score']}%")
    c4.metric("Embedding", f"{result['embedding_score']}%" if result["embedding_score"] is not None else "off")
    st.progress(min(int(result["final_score"]), 100))

    left, right = st.columns(2)
    with left:
        st.subheader("✅ Matched skills")
        st.write(", ".join(result["matched"]) or "None found")
        st.subheader("❌ Missing skills")
        st.write(", ".join(result["missing"]) or "None, great!")
    with right:
        st.subheader("Matched vs missing")
        chart = pd.DataFrame(
            {"count": [len(result["matched"]), len(result["missing"])]},
            index=["Matched", "Missing"],
        )
        st.bar_chart(chart)

    st.subheader("Candidate details (NER + regex)")
    st.json(info)

    with st.expander("Resume with matched skills highlighted"):
        highlighted = resume
        for s in sorted(result["matched"], key=len, reverse=True):
            highlighted = re.sub(
                r"(?<![\w+#])(" + re.escape(s) + r")(?![\w+#])",
                r"**\1**", highlighted, flags=re.IGNORECASE,
            )
        st.markdown(highlighted.replace("\n", "  \n"))

    st.subheader("💡 Suggestions")
    with st.spinner("Generating feedback..."):
        st.markdown(get_feedback(resume, jd, result["missing"], api_key or None))
