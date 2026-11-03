# HN-Sage: Semantic Search & RAG over Hacker News

**HN-Sage** is a full-stack, AI-powered search engine that goes beyond keyword matching. It allows you to ask natural language questions (e.g., *"What are people building with Rust lately?"*) and uses a local Large Language Model to synthesize answers based on live Hacker News discussions, complete with source citations.

The project uses a **RAG (Retrieval-Augmented Generation)** architecture to ensure the AI answers are grounded in real, up-to-date data rather than hallucinated.

## 🚀 Features

- **Hybrid Frontend:** Toggle seamlessly between traditional keyword filtering and AI-powered semantic search.
- **Vector Search:** Uses `sentence-transformers` to embed the meaning of stories and `FAISS` to perform lightning-fast semantic retrieval.
- **100% Local AI:** Powered by **Ollama (Llama 3.2)**. No API keys required, no data leaves your machine, and zero inference costs.
- **Source Citations:** The AI cites its sources using `[1]`, `[2]`, etc., and displays the retrieved Hacker News links below the answer.
- **Clean UI:** A modern, responsive, light/dark-mode enabled interface built with vanilla HTML/CSS/JS.

## 🧠 How the RAG Pipeline Works

1. **Ingestion (`ingest.py`):** Fetches top stories from the official Hacker News API. It extracts the title and text, and uses a Sentence Transformer (`all-MiniLM-L6-v2`) to convert them into dense vector embeddings.
2. **Indexing:** The embeddings are stored in a local `FAISS` vector database (`hn.faiss`), and the story metadata is stored in `SQLite` (`hn.db`).
3. **Retrieval (`server.py`):** When a user asks a question, the query is embedded into the same vector space. FAISS performs a cosine similarity search to retrieve the top 5 most relevant stories.
4. **Generation:** The retrieved stories are injected into a prompt context window and sent to the local **Llama 3.2** model via Ollama. The model is instructed to answer *only* using the provided context.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Machine Learning:** Sentence-Transformers (Hugging Face), FAISS (Facebook AI Similarity Search)
- **LLM Engine:** Ollama (Llama 3.2)
- **Frontend:** HTML5, Vanilla JavaScript, CSS3
- **Database:** SQLite

## 💻 How to Run Locally

### Prerequisites
- Python 3.9+
- [Ollama installed](https://ollama.com/) on your machine.

### 1. Download the AI Model
Open a terminal and pull the Llama model:
```bash
ollama run llama3.2