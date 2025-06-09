# SHARP-Application

**Voice-Driven Multilingual RAG Assistant with AI Avatar**

---

## Steps to run the project - run the commands in terminal

1. git clone https://github.com/vipularya135/SHARP-Application
2. cd SHARP-Application
3. pip install -r requirements.txt
4. streamlit run streamlit-Text-RAG.py
---

## Overview

SHARP-Application is a real-time, voice-first virtual technician that assists users in troubleshooting consumer appliances (e.g., washing machines) by engaging in natural conversations in their native language. Powered by Retrieval-Augmented Generation (RAG), the system retrieves precise instructions from multilingual user manuals and delivers responses through an animated AI avatar, providing a seamless, human-like customer support experience.

---

## Objectives

- **Voice-First Interaction:** Users can ask troubleshooting questions by speaking naturally in any supported language (e.g., English, Japanese, Chinese, Hindi).
- **Accurate Retrieval:** Employ RAG techniques to search and extract relevant manual sections for user queries.
- **Multilingual Understanding:** Detect spoken language, process queries, and generate responses in the same language.
- **Realistic Avatar Response:** Present answers via a lifelike AI avatar with synchronized speech and facial movements, enhancing engagement.

---

## Key Features

- **Automatic Speech Recognition (ASR):**  
  Captures live audio, transcribes speech, and identifies language.

- **RAG-Based Knowledge Access:**  
  Ingests and indexes multilingual manual content as vector embeddings. Retrieves top-k relevant passages for user queries.

- **Large Language Model (LLM) Response Generation:**  
  Combines retrieved context with user’s question to generate concise, accurate answers.

- **Text-to-Speech (TTS) Synthesis:**  
  Converts generated text into natural-sounding speech in the detected language.

- **AI Avatar Renderer:**  
  Uses viseme-driven lip syncing and facial animation to deliver responses through a virtual assistant graphic.

---

## Technical Architecture

- **Frontend:** React-based web application with WebRTC for audio streaming and Three.js (or D-ID) for avatar rendering.
- **ASR & Language ID:** OpenAI Whisper (or alternative) for real-time speech-to-text and language detection.
- **Vector Database:** ChromaDB (or Pinecone) to store and search embeddings of manual chunks.
- **RAG Pipeline:** LangChain to handle document ingestion, splitting, embedding, and retrieval.
- **LLM Backend:** Google Gemini Pro for answer generation.
- **TTS Engine:** ElevenLabs or Google Cloud TTS for multilingual speech synthesis.
- **Orchestration API:** FastAPI to coordinate ASR, retrieval, LLM calls, and TTS.

---

## Benefits

- **24/7 On-Demand Support:** Users receive instant guidance without waiting for a human technician.
- **User Satisfaction:** A natural conversational interface increases accessibility and ease of use.

---

## Steps to Start the Application

### 1. Data Collection
- Gather user manuals for various consumer products in multiple languages.

### 2. Simple Text RAG
- Implement a basic RAG pipeline:
  - User inputs a text query.
  - System retrieves and responds in text.
- Tools: [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Pinecone](https://www.pinecone.io/), [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings).

### 3. Integrate Free ASR and TTS APIs
- Research and select free/open-source tools for ASR and TTS:
  - **ASR:** [OpenAI Whisper](https://github.com/openai/whisper), [Vosk](https://alphacephei.com/vosk/).
  - **TTS:** [Coqui TTS](https://github.com/coqui-ai/TTS), [Google Cloud TTS (free tier)](https://cloud.google.com/text-to-speech), [ElevenLabs](https://elevenlabs.io/).

### 4. Voice Output RAG
- Extend text RAG to output responses as voice using a TTS engine.

### 5. Voice Input RAG
- Enable voice queries using ASR.
- Process spoken questions and respond with both text and voice.

### 6. Multilingual Response Logic
- Detect user’s spoken language automatically.
- Ensure the system replies in the same language as the input.

### 7. Avatar Integration (Optional for MVP)
- Use Three.js or [D-ID API](https://www.d-id.com/) to render an animated avatar that lip syncs with the generated speech.

---

## Usage Flow

1. **User**: Speaks a troubleshooting question in their language.
2. **System**:  
   a. Captures and transcribes the speech (ASR + Language ID).  
   b. Searches manuals for relevant troubleshooting steps (RAG).  
   c. Generates a concise answer (LLM).  
   d. Converts the answer to speech (TTS).  
   e. Animates the AI avatar to deliver the answer with synchronized lips and facial movement.

