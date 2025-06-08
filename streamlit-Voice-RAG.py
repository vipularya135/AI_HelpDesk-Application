# ——— Patch 1: Stop Streamlit watcher hitting torch._classes.__path__ ———
import torch
class _DummyPath:
    _path = []
torch._classes.__path__ = _DummyPath()

# ——— Patch 2: Make SentenceTransformer.to() fall back to to_empty() on meta modules ———
import sentence_transformers as _st
_BaseST = _st.SentenceTransformer
class SentenceTransformer(_BaseST):
    def to(self, *args, **kwargs):
        try:
            return super().to(*args, **kwargs)
        except NotImplementedError:
            # fallback for meta‐tensors
            return super().to_empty(*args, **kwargs)

# ——— Your usual imports ———
import streamlit as st
import PyPDF2
import numpy as np
from typing import List
from langdetect import detect, detect_langs
# note: we no longer import SentenceTransformer here, we use our patched one above
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
import tempfile
import os


class RAGSingleLanguage:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        # uses patched SentenceTransformer
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
        q_en = self.translate(question, 'en')
        q_emb = self.embedder.encode([q_en], convert_to_numpy=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top = np.argsort(sims)[::-1][:3]
        ctx = "\n\n".join(self.chunks[i] for i in top)
        prompt = (
            "Answer the question using only the context below. "
            "Do not hallucinate.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q_en}"
        )
        try:
            out = self.model.generate_content(prompt).text.strip()
        except Exception as e:
            return f"Error: {e}"
        return self.translate(out, self.language)


def recognize_voice(language='en-IN') -> str:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        st.info("Listening... Speak now.")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.warning("⏰ No speech detected (timeout). Try again.")
            return ""
    try:
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        st.error("❗Could not understand audio. Please try again.")
    except sr.RequestError as e:
        st.error(f"🚫 Could not request results; check your internet. Error: {e}")
    return ""


def speak_text(text: str, lang_code='en'):
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")


def main():
    st.set_page_config(page_title="Multilingual QA", page_icon="📚")
    st.title("📚 Voice-Driven Multilingual RAG Assistant")
    api_key = os.getenv('GENAI_API_KEY', 'AIzaSyA5xtoT9HAjH-wsa7OHFXlBjRRcXwCFBMg')
    if 'rag' not in st.session_state:
        st.session_state.rag = RAGSingleLanguage(api_key)
        st.session_state.file_done = False
        st.session_state.langs = []
        st.session_state.lang = None
        st.session_state.voice_question = ""

    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        "1. Upload PDF\n"
        "2. Pick language\n"
        "3. Ask questions by typing or voice\n"
        "4. Get spoken answers"
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

        st.markdown("**Option 1: Type your question**")
        typed_question = st.text_input(f"Ask in {sel}:")
        question = typed_question or st.session_state.voice_question

        st.markdown("**Option 2: Or use voice input**")
        if st.button("🎙️ Speak Your Question"):
            lang_code = st.session_state.lang + "-IN" if st.session_state.lang == "en" else st.session_state.lang
            recognized = recognize_voice(lang_code)
            if recognized:
                st.session_state.voice_question = recognized
                st.success(f"🎤 You said: {recognized}")
            else:
                st.warning("⚠️ No valid speech recognized.")

        if st.button("Get Answer") and question:
            st.markdown(f"🔍 Question: `{question}`")
            with st.spinner("Thinking..."):
                ans = st.session_state.rag.answer_question(question)
                st.markdown(f"**Answer ({sel}):** {ans}")
                speak_text(ans, lang_code=st.session_state.lang)
                st.session_state.voice_question = ""  # clear voice question after use


if __name__ == '__main__':
    main()
