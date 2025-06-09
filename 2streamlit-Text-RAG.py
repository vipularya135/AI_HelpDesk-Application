import streamlit as st
import PyPDF2
import numpy as np
from typing import List
from langdetect import detect, detect_langs
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

class RAGSingleLanguage:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.language: str = 'en'

    def detect_languages(self, text: str) -> List[str]:
        segment_size = 1000
        segments = [text[i:i + segment_size] for i in range(0, len(text), segment_size)]
        lang_probs = {}
        for seg in segments:
            try:
                for lang in detect_langs(seg):
                    lang_probs.setdefault(lang.lang, 0.0)
                    lang_probs[lang.lang] = max(lang_probs[lang.lang], lang.prob)
            except:
                continue
        result = [lang for lang, prob in lang_probs.items() if prob >= 0.2]
        return result if result else ['en']

    def translate(self, text: str, target_lang: str) -> str:
        try:
            source = detect(text)
        except:
            source = 'en'
        if source.lower() == target_lang.lower():
            return text
        prompt = (f"Translate the following text to '{target_lang.upper()}'. "
                  "Return only the translated text:\n\n" + text)
        try:
            resp = self.model.generate_content(prompt)
            return resp.text.strip()
        except:
            return text

    def process_document(self, uploaded_file, chunk_size: int = 500) -> List[str]:
        if uploaded_file is None:
            raise FileNotFoundError("No file uploaded")
        raw = []
        reader = PyPDF2.PdfReader(uploaded_file)
        for p in reader.pages:
            raw.append(p.extract_text() or '')
        full = ' '.join(raw)
        words = full.split()
        self.chunks = [' '.join(words[i:i + chunk_size]) 
                       for i in range(0, len(words), chunk_size)]
        self.embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)
        return self.detect_languages(full)

    def set_language(self, lang: str):
        self.language = lang

    def answer_question(self, question: str) -> str:
        # 1. Translate question into English for retrieval
        q_en = self.translate(question, 'en')
        # 2. Find top-3 relevant chunks
        q_emb = self.embedder.encode([q_en], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top = np.argsort(sims)[::-1][:3]
        ctx = "\n\n".join(self.chunks[i] for i in top)
        # 3. Ask Gemini
        prompt = (
            "Answer the question using only the context below. "
            "Do not hallucinate.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q_en}"
        )
        try:
            out = self.model.generate_content(prompt).text.strip()
        except Exception as e:
            return f"Error: {e}"
        # 4. Translate back to user language
        return self.translate(out, self.language)

def main():
    st.set_page_config(page_title="Multilingual QA", page_icon="📚")
    st.title("📚 PDF Q&A – Multilingual")

    # Hard-coded key
    api_key = "AIzaSyDV512NPzGh19E9gPGGLn83f64CYw6lPzQ"

    # Session state
    if 'rag' not in st.session_state:
        st.session_state.rag = RAGSingleLanguage(api_key)
        st.session_state.file_done = False
        st.session_state.langs = []
        st.session_state.lang = None

    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        "1. Upload PDF\n"
        "2. Pick language\n"
        "3. Ask questions\n"
        "4. Get answers in same language"
    )

    uploaded = st.file_uploader("Upload your PDF manual", type="pdf")
    if uploaded and not st.session_state.file_done:
        with st.spinner("Processing PDF..."):
            try:
                st.session_state.langs = st.session_state.rag.process_document(uploaded)
                st.session_state.file_done = True
                st.success("✅ Document processed!")
            except Exception as e:
                st.error(f"Failed: {e}")
                return

    if st.session_state.file_done:
        sel = st.selectbox("Select language:", options=[l.upper() for l in st.session_state.langs])
        if sel:
            st.session_state.lang = sel.lower()
            st.session_state.rag.set_language(st.session_state.lang)

        question = st.text_input(f"Ask your question in {sel}:")
        if st.button("Get Answer") and question:
            with st.spinner("Thinking..."):
                ans = st.session_state.rag.answer_question(question)
                st.markdown(f"**Answer ({sel}):**  {ans}")

if __name__ == '__main__':
    main()
