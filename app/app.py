import streamlit as st
import sys
import os

# Add the parent directory to the path so we can import predict.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import summarize

st.set_page_config(page_title="Text Summarizer", layout="wide", page_icon="📝")

st.title("📰 AI News Article Summarizer")
st.markdown(
    """
    **Welcome!** Paste a long news article below and get a concise 2-3 sentence summary instantly.
    *Assumes Ollama is running locally with the `summarizer` model.*
    """
)

article = st.text_area("📄 Article", height=300, placeholder="Paste your article here...")

if st.button("✨ Summarize", type="primary"):
    if article.strip():
        with st.spinner("Generating summary..."):
            summary = summarize(article)
        
        st.subheader("📝 Summary")
        if "Error connecting" in summary:
            st.error(summary)
        else:
            st.info(summary)
    else:
        st.warning("Please paste an article first.")
