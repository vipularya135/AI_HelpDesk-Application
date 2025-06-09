import os
import numpy as np
import PyPDF2
from typing import List
from langdetect import detect, detect_langs
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


class RAGSingleLanguage:
    def __init__(self, api_key: str):
        # Configure Google Generative AI
        genai.configure(api_key='AIzaSyDV512NPzGh19E9gPGGLn83f64CYw6lPzQ')
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.language: str = 'en'

    def detect_languages(self, text: str) -> List[str]:
        """
        Split the document into segments and detect languages per segment.
        Return unique languages with probability > 0.2.
        """
        # Break text into manageable segments for detection
        segment_size = 1000
        segments = [text[i:i + segment_size] for i in range(0, len(text), segment_size)]
        lang_probs = {}
        for seg in segments:
            try:
                for lang in detect_langs(seg):
                    lang_probs.setdefault(lang.lang, 0.0)
                    # accumulate max probability seen for this lang
                    lang_probs[lang.lang] = max(lang_probs[lang.lang], lang.prob)
            except Exception:
                continue

        # Keep languages above threshold
        detected = [lang for lang, prob in lang_probs.items() if prob >= 0.2]
        return detected if detected else ['en']

    def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text to the target language using the GenAI model.
        If text is already in target, return as-is.
        """
        try:
            source_lang = detect(text)
        except:
            source_lang = 'en'

        if source_lang.lower() == target_lang.lower():
            return text

        prompt = (
            f"Translate the following text to '{target_lang.upper()}'."
            " Only return the translated text.\n\n" + text
        )
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def process_document(self, pdf_path: str, chunk_size: int = 500) -> List[str]:
        """
        Read PDF, split into word-chunks, generate embeddings,
        and detect available languages in the document.
        Returns list of detected language codes.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found at: {pdf_path}")

        # Extract raw text
        text = []
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                content = page.extract_text() or ''
                text.append(content)
        full_text = ' '.join(text).strip()

        # Chunk by words
        words = full_text.split()
        self.chunks = [ ' '.join(words[i:i+chunk_size])
                        for i in range(0, len(words), chunk_size) ]

        # Compute embeddings once
        self.embeddings = self.embedder.encode(self.chunks, convert_to_numpy=True)

        # Detect languages
        return self.detect_languages(full_text)

    def set_language(self, lang: str):
        """Set the RAG system's output language."""
        self.language = lang

    def answer_question(self, user_question: str) -> str:
        """
        Given a user question, translate to English, retrieve relevant contexts,
        generate answer, then translate back to user language.
        """
        # Translate question to English for embedding
        question_en = self.translate(user_question, 'en')

        # Compute similarity with document chunks
        query_emb = self.embedder.encode([question_en], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:3]
        context = '\n\n'.join(self.chunks[i] for i in top_indices)

        # Build prompt for GenAI
        prompt = (
            "Answer the question using only the context below. "
            "Do not hallucinate any information.\n\n"
            f"Context:\n{context}\n\nQuestion: {question_en}"
        )
        response = self.model.generate_content(prompt)
        answer_en = response.text.strip()

        # Translate answer back to user's language
        return self.translate(answer_en, self.language)


def select_language(detected_languages: List[str]) -> str:
    print("\n📚 Detected languages in document:")
    for idx, lang in enumerate(detected_languages, start=1):
        print(f"{idx}. {lang.upper()}")
    while True:
        try:
            choice = int(input("\nSelect language number to continue: "))
            if 1 <= choice <= len(detected_languages):
                return detected_languages[choice - 1]
        except ValueError:
            pass
        print("❌ Invalid choice. Try again.")


def main():
    pdf_path = r"C:\Users\krish\Desktop\sharp\documents\SHARP_FRIDGE.pdf"
    api_key = os.getenv('GENAI_API_KEY', 'YOUR_API_KEY_HERE')

    rag = RAGSingleLanguage(api_key)
    print("🔍 Processing document...")
    available_langs = rag.process_document(pdf_path)

    chosen = select_language(available_langs)
    rag.set_language(chosen)
    print(f"\n✅ You selected: {chosen.upper()}. Ask questions in this language now.")

    while True:
        question = input(f"\nAsk your question in {chosen.upper()} (or type 'quit'): ")
        if question.strip().lower() == 'quit':
            break
        answer = rag.answer_question(question)
        print(f"\n🧠 Answer ({chosen.upper()}):\n{answer}\n")


if __name__ == '__main__':
    main()