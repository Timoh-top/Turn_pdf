import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Helper: Extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Helper: Split text into clean sentences
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip().replace('\n', ' ') for s in sentences if len(s.strip()) > 20]

# Helper: TF-IDF extractive summarizer with bullet points
def summarize_text(text, num_sentences=10):
    sentences = split_into_sentences(text)
    if len(sentences) <= num_sentences:
        return "\n".join(f"- {s}" for s in sentences)

    vect = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vect.fit_transform(sentences)
    scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
    top_indices = scores.argsort()[-num_sentences:][::-1]
    summary_sentences = [sentences[i] for i in sorted(top_indices)]
    bullet_summary = "\n".join(f"- {s}" for s in summary_sentences)
    return bullet_summary

# Helper: Chunk text for retrieval
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Helper: Retrieve relevant chunk based on question
def retrieve_answer(question, chunks):
    vect = TfidfVectorizer(stop_words='english')
    vectors = vect.fit_transform(chunks + [question])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
    idx = np.argmax(cosine_sim)
    return chunks[idx]

# ------------------ Streamlit App UI ------------------

# Page config
st.set_page_config(
    page_title="📄 PDF Summarizer + Q&A",
    page_icon="📄",
    layout="wide"
)

# Sidebar
st.sidebar.title("📄 PDF Summarizer + Q&A App")
st.sidebar.info("""
🚀 **Instructions:**

1️⃣ Upload a **text-based PDF**  
2️⃣ Click **Generate Summary**  
3️⃣ Ask **questions** about your PDF

All **API-free, zero-cost** for your learning and Agentic AI practice.
""")

uploaded_file = st.sidebar.file_uploader("📥 Upload your PDF", type=["pdf"])

num_sentences = st.sidebar.slider(
    "🔹 Number of Bullet Points in Summary",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

st.markdown("<h1 style='text-align: center;'>📄 Free PDF Summarizer + Q&A App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>⚡ Built for your zero-cost Agentic AI projects, learning, and sharing.</p>", unsafe_allow_html=True)

if uploaded_file:
    st.success("✅ PDF uploaded successfully. Extracting text...")
    raw_text = extract_text_from_pdf(uploaded_file)

    if not raw_text.strip():
        st.error("❌ No extractable text found. Please upload a PDF with selectable text.")
        st.stop()

    st.info(f"✅ Extracted **{len(raw_text)} characters** from your PDF.")

    with st.expander("📑 Generate Bullet-Point Summary", expanded=True):
        if st.button("Generate Summary"):
            with st.spinner("Generating clean bullet-point summary..."):
                summary = summarize_text(raw_text, num_sentences=num_sentences)
            st.subheader("✨ Bullet-Point Summary")
            st.markdown(summary)

    chunks = chunk_text(raw_text)

    with st.expander("💬 Ask Questions About Your PDF", expanded=True):
        question = st.text_input("Ask a question about your PDF:")
        if question:
            with st.spinner("Retrieving the best matching section..."):
                answer = retrieve_answer(question, chunks)
            st.success("✅ Retrieved Section:")
            st.write(answer)

else:
    st.info("👈 Upload a PDF from the **sidebar** to get started.")

st.markdown("---")
st.caption("⚡ Built by Timoh-top for Agentic AI practice, scholarships, and your learning portfolio.")
