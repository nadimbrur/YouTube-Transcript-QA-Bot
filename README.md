# 📺 YouTube Transcript QA Bot (LangChain + Hugging Face + Streamlit)

This is a simple app that allows you to ask questions about a YouTube video by analyzing its transcript using embeddings and a large language model. It uses:
- `LangChain` for chaining LLM calls,
- `FAISS` for semantic search,
- `Hugging Face` models (like Zephyr-7B) for generation,
- `Streamlit` for the web interface.

---

## 🚀 Features

- Extracts transcript from any YouTube video (with captions).
- Splits and embeds transcript using Sentence Transformers.
- Stores and retrieves relevant chunks with FAISS.
- Answers your question using a Hugging Face-hosted LLM.
- Lightweight and runs entirely client-side with free tier APIs.

---

## 🛠️ Requirements

### Python 3.8+

Install dependencies:

```bash
pip install -r requirements.txt
```

## 💻 How to Run
streamlit run yt_chatbot.py

📋 Usage
Paste a YouTube Video Link (e.g., Gfr50f6ZBvo).

Wait for the transcript to be fetched and embedded.

Enter a question about the video content.

Get answers directly sourced from the transcript!

## 📌 Notes
YouTube must have English captions enabled.

The Hugging Face free tier provides limited monthly usage (about 1K calls).

Transcript size and LLM response may vary with model size and context limits.

## 🧠 Example Models
Embedding: sentence-transformers/all-MiniLM-L6-v2

Generation: HuggingFaceH4/zephyr-7b-beta

You can swap these for other models in Hugging Face as long as they are Inference API-compatible.