import streamlit as st
import pdfplumber
import requests
import re
import textwrap

# Setup
st.set_page_config(
    page_title="📄 Hugging Face PDF Summarizer + Q&A",
    page_icon="📄",
    layout="wide"
)

# Hugging Face Inference API call
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}

def query_hf_summarization(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.json()

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Chunk text into smaller parts for summarization
def chunk_text_for_summary(text, max_tokens=1024):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Sidebar
st.sidebar.title("📄 PDF Summarizer (Hugging Face)")
st.sidebar.info("""
🚀 **Steps:**
1️⃣ Upload a **text-based PDF**  
2️⃣ Click **Generate Summary**  
⚡ Uses Hugging Face API for **clean abstractive summaries**
""")

uploaded_file = st.sidebar.file_uploader("📥 Upload your PDF", type=["pdf"])

max_chunks = st.sidebar.slider("🔹 Max chunks to summarize (limit for free tier):", 1, 10, 3)

st.markdown("<h1 style='text-align: center;'>📄 Hugging Face PDF Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>✨ Powered by Hugging Face Inference API | Built for your Agentic AI projects</p>", unsafe_allow_html=True)

if uploaded_file:
    st.success("✅ PDF uploaded. Extracting text...")
    raw_text = extract_text_from_pdf(uploaded_file)
    if not raw_text.strip():
        st.error("❌ No extractable text found. Please upload a valid, text-based PDF.")
        st.stop()
    st.info(f"✅ Extracted **{len(raw_text)} characters** from your PDF.")

    with st.expander("📑 Generate Abstractive Summary", expanded=True):
        if st.button("Generate Summary"):
            with st.spinner("Summarizing using Hugging Face..."):
                chunks = chunk_text_for_summary(raw_text)
                summaries = []
                for idx, chunk in enumerate(chunks[:max_chunks]):
                    try:
                        output = query_hf_summarization(chunk)
                        if isinstance(output, list) and "summary_text" in output[0]:
                            summary_text = output[0]["summary_text"]
                            wrapped_summary = textwrap.fill(summary_text, width=100)
                            summaries.append(f"**Chunk {idx+1}:**\n{wrapped_summary}\n")
                        else:
                            summaries.append(f"**Chunk {idx+1}:**\n❌ Could not summarize this chunk.\n")
                    except Exception as e:
                        summaries.append(f"**Chunk {idx+1}:**\n❌ Error: {e}\n")
                final_summary = "\n\n".join(summaries)
            st.subheader("✨ Abstractive Summary")
            st.markdown(final_summary)

else:
    st.info("👈 Upload a PDF from the **sidebar** to get started.")

st.markdown("---")
st.caption("⚡ Built by Timothy Ajewole for Agentic AI learning, scholarships, and portfolio.")
