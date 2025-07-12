import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Helper: Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
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

# Helper: Summarize using extractive preview
def summarize_text(text, max_chars=1500):
    return text[:max_chars] + ("..." if len(text) > max_chars else "")

# Helper: Retrieve relevant chunk based on question
def retrieve_answer(question, chunks):
    vect = TfidfVectorizer(stop_words='english')
    vectors = vect.fit_transform(chunks + [question])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
    idx = np.argmax(cosine_sim)
    return chunks[idx]

# Streamlit App
st.title("Free PDF Summarizer + Q&A App")
st.markdown("""
**How to use:**
Upload a **text-based PDF** (not a scanned image).  
Click **Show Extractive Summary** to preview the PDF.  
Enter your question to get a retrieval-based answer from your PDF.
""")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded successfully.")
    raw_text = extract_text_from_pdf(uploaded_file)
    if not raw_text.strip():
        st.error("❌ No extractable text found in this PDF. Please try another PDF with selectable text.")
        st.stop()
    else:
        st.info(f"PDF extracted with {len(raw_text)} characters.")

        # Show summary
        if st.button("Show Extractive Summary"):
            summary = summarize_text(raw_text)
            st.subheader("📑 Summary Preview")
            st.write(summary)

        # Chunk for retrieval
        chunks = chunk_text(raw_text)

        # Q&A
        question = st.text_input("💬 Ask a question about your PDF:")
        if question:
            with st.spinner("Retrieving the best matching answer..."):
                answer = retrieve_answer(question, chunks)
            st.success("Answer (retrieved section):")
            st.write(answer)

st.markdown("---")
st.caption("🚀 Built for zero-cost learning and Agentic AI practice. Deployable on Streamlit Cloud for others to use.")
