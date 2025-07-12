import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.summarization import summarize

# Helper: Extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Helper: Chunk text for retrieval
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Helper: Summarize using gensim
def summarize_text(text, ratio=0.1):
    try:
        summary = summarize(text, ratio=ratio, split=False)
        if not summary.strip():
            return text[:1500] + ("..." if len(text) > 1500 else "")
        return summary
    except ValueError:
        return text[:1500] + ("..." if len(text) > 1500 else "")

# Helper: Retrieve relevant chunk based on question
def retrieve_answer(question, chunks):
    vect = TfidfVectorizer(stop_words='english')
    vectors = vect.fit_transform(chunks + [question])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
    idx = np.argmax(cosine_sim)
    return chunks[idx]

# Streamlit App UI
st.title("📄 Free PDF Summarizer + Q&A App")
st.markdown("""
🚀 **Welcome!** This app allows you to:
1️⃣ Upload a **text-based PDF**  
2️⃣ Generate a **clean extractive summary**  
3️⃣ **Ask questions** to retrieve relevant sections from your PDF

⚡ Fully free for your learning, Agentic AI practice, and portfolio building.
""")

uploaded_file = st.file_uploader("📥 Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully. Extracting text...")
    raw_text = extract_text_from_pdf(uploaded_file)
    if not raw_text.strip():
        st.error("❌ No extractable text found in this PDF. Please try another PDF with selectable text.")
        st.stop()
    else:
        st.info(f"✅ Extracted {len(raw_text)} characters from your PDF.")

        # Show summary
        if st.button("📑 Generate Extractive Summary"):
            with st.spinner("Generating summary..."):
                summary = summarize_text(raw_text)
            st.subheader("📑 Summary Preview")
            st.write(summary)

        # Chunk for retrieval
        chunks = chunk_text(raw_text)

        # Q&A
        question = st.text_input("💬 Ask a question about your PDF:")
        if question:
            with st.spinner("Retrieving the best matching section..."):
                answer = retrieve_answer(question, chunks)
            st.success("✅ Retrieved Section:")
            st.write(answer)

st.markdown("---")
st.caption("⚡ Built by Timoh-top for zero-cost Agentic AI learning and portfolio projects.")
