"""UOWD Student Compass: independently developed academic information-retrieval demo.

Pretrained MiniLM embeddings vs TF-IDF; 89 curated public-source passages.
No university login, personal student data, or generative answer fabrication.
"""
from __future__ import annotations

import re
import streamlit as st

from src.retrieval import MiniLMRetriever, TfidfRetriever, load_documents

st.set_page_config(page_title="UOWD Student Compass", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.0rem; max-width: 1100px;}
[data-testid="stMetric"] {background: #f3f7fc; border-radius: 12px; padding: 12px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def docs():
    return load_documents()

@st.cache_resource(show_spinner="Loading the selected retrieval model…")
def get_retriever(kind: str):
    return TfidfRetriever(docs()) if kind == "TF-IDF" else MiniLMRetriever(docs())

def is_private_record_question(q: str) -> bool:
    """Small targeted rule, not comprehensive private-information detection."""
    q = q.lower()
    personal = re.search(r"\b(my|mine)\b", q)
    private = re.search(r"\b(gpa|cgpa|grades?|marks?|transcripts?|attendance|student\s+id|password)\b", q)
    return bool(personal and private)

EXAMPLES = [
    "Is there a free UOWD bus from Dubai Internet City Metro?",
    "What scholarships are available to Master's students?",
    "What is the structure of the Master of Applied Artificial Intelligence?",
    "When is the Autumn 2026 withdrawal deadline without academic penalty?",
    "What is my university GPA?",
]

st.title("🎓 UOWD Student Compass")
st.caption("Independent university information retrieval prototype · Not affiliated with or endorsed by UOWD")
col1, col2, col3 = st.columns(3)
col1.metric("UOWD webpages", "21")
col2.metric("Curated facts", str(len(docs())))
col3.metric("Evaluation questions", "100")

with st.sidebar:
    st.header("Retrieval settings")
    method = st.radio("Choose a retrieval model", ["MiniLM", "TF-IDF"], index=0,
                      help="MiniLM uses 384-dimensional sentence embeddings; TF-IDF is the lexical baseline.")
    st.caption("MiniLM's first use downloads its public pretrained model. This is cached on the server.")
    st.divider()
    st.subheader("Measured benchmark")
    st.caption("Previous 100-query evaluation (not recomputed by this demo)")
    st.markdown("**Precision@1:** MiniLM 82% · TF-IDF 61%")
    st.markdown("**Recall@3:** MiniLM 97% · TF-IDF 80%")
    st.markdown("**MRR@5:** MiniLM 0.8967 · TF-IDF 0.7143")
    st.caption("One manually assigned gold passage per question; not chatbot accuracy.")

st.subheader("Ask a question")
example = st.selectbox("Try a demonstration question", ["Type my own question"] + EXAMPLES)
with st.form("search_form", clear_on_submit=False):
    question = st.text_input("Student question", value="" if example == "Type my own question" else example,
                             placeholder="e.g. Does UOWD provide a shuttle from the metro?")
    submitted = st.form_submit_button("🔎 Search UOWD information", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")
    elif is_private_record_question(question):
        st.warning("🔒 I cannot access your personal academic records or GPA. Please use the official UOWD student portal or contact Student Services.")
        st.caption("This is a targeted rule-based safeguard; it does not detect every out-of-scope question.")
    else:
        try:
            retriever = get_retriever(method)
            hits = retriever.search(question, top_k=3)
            if hits:
                best = hits[0]
                st.success(f"Top {method} match · {best['title']}")
                st.write(best["text"])
                st.link_button("Open official UOWD source", best["source_url"])
                st.caption(f"Passage {best['id']} · Similarity {best['score']:.3f} (not a probability) · Reviewed {best.get('verified_date','unspecified')}")
                st.markdown("#### Top three retrieved passages")
                for i, hit in enumerate(hits, 1):
                    with st.expander(f"{i}. {hit['title']} · similarity {hit['score']:.3f}", expanded=i == 1):
                        st.write(hit["text"])
                        st.markdown(f"[Official university webpage]({hit['source_url']})")
                        st.caption(f"ID: {hit['id']}")
                if best['score'] < (0.3 if method == 'MiniLM' else 0.12):
                    st.info("⚠️ Low similarity. The top result might not answer your question. Please verify against the university source.")
            else:
                st.warning("No matching information found in this curated collection.")
        except Exception as exc:
            st.error(f"The {method} model could not be loaded. Switch to TF-IDF for the backup demo. Error: {exc}")

with st.expander("About this NLP project"):
    st.write("This academic prototype ranks 89 manually curated, English-language facts from 21 public UOWD pages. "
             "TF-IDF compares lexical features; MiniLM embeds questions and passages using a pretrained Transformer. "
             "Both approaches return source-linked passages rather than generating new answers.")
    st.write("Limitations: no live website refresh, no private-record access, one assigned gold passage per evaluation question, "
             "unvalidated low-similarity warning, and no comprehensive out-of-scope rejection.")

st.caption("Academic prototype only. Verify dates, prices, and university policies on official UOWD webpages before relying on them.")
