
import os
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, List, Mapping, Any

# PDF text extraction
import fitz  # PyMuPDF

# Load .env if present
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="LangChain + Gemini Summarizer", layout="wide")
st.title("AI Text Summarizer ")

if not GEMINI_API_KEY:
    st.warning("GEMINI_API_KEY not found. Set environment variable GEMINI_API_KEY or create a .env file.")

# --- LLM wrapper for LangChain using google.generativeai ---
try:
    import google.generativeai as genai
    from langchain.llms.base import LLM
    from langchain import LLMChain, PromptTemplate

    genai.configure(api_key=GEMINI_API_KEY)

    class GoogleGemini(LLM):
        model: str = "gemini-2.0-flash"

        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            # Using the generative model API for Gemini 2.0 Flash
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)

            try:
                return response.text
            except Exception:
                try:
                    return response.candidates[0].content.parts[0].text
                except Exception:
                    return str(response)

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"model": self.model}

        @property
        def _llm_type(self) -> str:
            return "google-gemini-flash"

except Exception as e:
    GoogleGemini = None
    st.error("Could not import google.generativeai or langchain. Make sure dependencies are installed.")

# --- Prompt Template ---
SUMMARY_PROMPT = """
You are an expert summarizer. Summarize the text below clearly and concisely in about {max_sentences} sentences.
Text:
{input_text}
\nSummary:
"""

prompt = PromptTemplate(input_variables=["input_text", "max_sentences"], template=SUMMARY_PROMPT)

# --- Helper functions ---
def extract_text_from_pdf(file) -> str:
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    chunks = []
    for i in range(0, len(text), max_chars):
        chunks.append(text[i:i + max_chars])
    return chunks

# --- Streamlit Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.markdown("Using **Gemini 2.0 Flash** model 🚀")
    max_sentences = st.slider("Approx. summary length (sentences per chunk)", 1, 10, 3)
    chunking_enabled = st.checkbox("Enable chunked summarization for long documents", True)
    temperature = st.slider("Temperature (creativity)", 0.0, 1.0, 0.2)

# --- Main Interface ---
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input text or upload a file")
    txt = st.text_area("Paste text to summarize", height=300)

    uploaded = st.file_uploader("Or upload a .txt or .pdf file", type=["txt", "pdf"])
    if uploaded is not None and not txt.strip():
        if uploaded.name.endswith(".txt"):
            txt = uploaded.read().decode("utf-8")
        elif uploaded.name.endswith(".pdf"):
            txt = extract_text_from_pdf(uploaded)

    if st.button("Summarize"):
        if not txt.strip():
            st.warning("Please provide input text or upload a file.")
        elif GoogleGemini is None:
            st.error("Gemini LLM wrapper not available. Check setup.")
        else:
            llm = GoogleGemini()

            chain = LLMChain(llm=llm, prompt=prompt)

            # --- Handle chunked summarization ---
            if chunking_enabled and len(txt) > 4000:
                st.info("Large document detected — using chunked summarization.")
                chunks = chunk_text(txt)
                summaries = []
                for i, chunk in enumerate(chunks):
                    st.write(f"Processing chunk {i + 1}/{len(chunks)}...")
                    result = chain.run(input_text=chunk, max_sentences=max_sentences)
                    summaries.append(result)
                combined_text = "\n".join(summaries)
                final_summary = chain.run(input_text=combined_text, max_sentences=max_sentences + 2)
                result = final_summary
            else:
                result = chain.run(input_text=txt, max_sentences=max_sentences)

            st.subheader("Summary")
            st.write(result)

with col2:
    st.subheader("Tips")
    st.write("• Uses the Gemini 2.0 Flash model (fast + efficient).\n• Supports .txt and .pdf files.\n• Chunked summarization improves handling of long docs.")

st.markdown("---")
st.caption("This Streamlit app uses LangChain + Gemini 2.0 Flash with PDF support and chunked summarization.")