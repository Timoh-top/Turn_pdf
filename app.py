import streamlit as st
import pdfplumber
import requests
import re
import textwrap

# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="📄 Hugging Face PDF Summarizer + Q&A",
    page_icon="📄",
    layout="wide"
)

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}

# ---------------------- HELPERS ----------------------

# Call Hugging Face summarization API
def query_hf_summarization(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    print(response.text)  # DEBUG: See raw API output in Streamlit logs
    if response.status_code != 200:
        return {"error": f"Status Code: {response.status_code}, {response.text}"}
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

# Split text into smaller chunks
def chunk_text_for_summary(text, max_tokens=500):
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

# ---------------------- STREAMLIT APP ----------------------

# Sidebar
st.sidebar.title("📄 PDF Summarizer (Hugging Face)")
st.sidebar.info("""
🚀 **Steps:**
1️⃣ Upload a **text-based PDF**  
2️⃣ Click **Generate Summary**  
⚡ Uses Hugging Face API for **clean abstractive summaries**
""")

uploaded_file = st.sidebar.file_uploader("📥 Upload your PDF", type=["pdf"])
max_chunks = st.sidebar.slider("🔹 Max chunks to summarize:", 1, 10, 3)

# Reset session state for new PDF uploads
if 'prev_file' not in st.session_state:
    st.session_state.prev_file = None

if uploaded_file != st.session_state.prev_file:
    st.session_state.prev_file = uploaded_file
    st.session_state['summary'] = ""

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
                    output = query_hf_summarization(chunk)
                    if isinstance(output, list) and "summary_text" in output[0]:
                        summary_text = output[0]["summary_text"]
                        wrapped_summary = textwrap.fill(summary_text, width=100)
                        summaries.append(f"**🔹 Chunk {idx+1}:**\n\n{wrapped_summary}\n")
                    else:
                        summaries.append(f"**🔹 Chunk {idx+1}:**\n❌ Could not summarize.\n\nError: {output}\n")
                final_summary = "\n\n".join(summaries)
                st.session_state['summary'] = final_summary

    if st.session_state.get('summary'):
        st.subheader("✨ Abstractive Summary")
        st.markdown(st.session_state['summary'])

else:
    st.info("👈 Upload a PDF from the **sidebar** to get started.")

st.markdown("---")
st.caption("⚡ Built by Timothy Ajewole for Agentic AI learning, and portfolio.")
