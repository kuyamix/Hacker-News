<div align="center">

# 🔥 HN-Sage
**Next-Generation Semantic Search & Retrieval-Augmented Generation (RAG) for Hacker News**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-FF6F00?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

*Ask natural language questions about live tech discussions, powered entirely by a local, privacy-first AI pipeline.*
</div>

---

## 📖 Overview

**HN-Sage** transforms the traditional Hacker News reading experience into an interactive, conversational interface. While traditional platforms rely on strict keyword matching, HN-Sage leverages dense vector embeddings to understand the *semantic meaning* behind your queries.

By combining **FAISS** for high-speed vector retrieval with **Llama 3.2** (via Ollama) for zero-shot text synthesis, HN-Sage acts as a fully local, zero-cost, hallucination-mitigated research assistant. It retrieves relevant discussions, synthesizes a coherent answer, and rigorously cites its sources.

## 🏗️ System Architecture

The application utilizes a decoupled client-server architecture, prioritizing high-performance ML inference on the backend and a lightweight, responsive client on the frontend.

```mermaid
graph TD
    A[User Query] -->|HTTP POST| B(FastAPI Backend)
    B -->|Encode Query| C[Sentence-Transformer: MiniLM]
    C -->|Asymmetric Vector Search| D[(FAISS Index)]
    D -->|Top-K Relevant Stories| B
    B -->|Construct Context Prompt| E[Ollama: Llama 3.2]
    E -->|Synthesized Answer| B
    B -->|JSON: Answer + Citations| F[Vanilla JS Frontend]
    F -->|Render UI| A
✨ Core Features
🧠 True Semantic Retrieval: Uses all-MiniLM-L6-v2 to embed text into a 384-dimensional space, capturing intent rather than just lexical overlap.

🔒 100% Privacy-First AI: No external API calls for generation. The entire RAG pipeline (retrieval + generation) runs offline on your local machine via Ollama.

📚 Verifiable Citations: The prompt engineering forces the LLM to adopt a strict citation format ([1], [2]), which the frontend dynamically maps back to live Hacker News URLs.

⚡ High-Performance Indexing: Utilizes FAISS (Facebook AI Similarity Search) with Inner Product (IP) metrics for sub-millisecond retrieval across the dataset.

🎨 Premium Hybrid UI: A custom-built, dependency-free frontend featuring dark/light mode, keyboard navigation (/ to focus, R to reload), and skeleton loaders.

🛠️ Technical Stack
Domain	Technologies
LLM & Inference	Ollama, Llama 3.2 (3B Parameters)
Vector Database	FAISS (IndexFlatIP)
Embeddings	Hugging Face sentence-transformers
Backend API	Python, FastAPI, Uvicorn, Pydantic
Data Persistence	SQLite3
Frontend	HTML5, CSS3, Vanilla JavaScript (ES6+)
🚀 Quickstart Guide
Prerequisites
Python 3.9+

Ollama Download and install here.

1. Initialize the Local AI Engine
Download the Llama 3.2 model to your local machine:

bash
ollama run llama3.2
(Once downloaded, type /bye to exit. Ollama will run silently in the background).

2. Configure the Environment
Clone the repository and set up a virtual environment:

bash
git clone https://github.com/YOUR_USERNAME/hn-sage.git
cd hn-sage

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
3. Ingest & Embed Data
Run the data pipeline to fetch the latest Hacker News and build the vector database:

bash
python ingest.py
(This generates hn.db, hn.faiss, and ids.json locally).

4. Launch the Application
Start the FastAPI backend and the static file server in two separate terminals.

Terminal 1 (AI Backend):

bash
uvicorn server:app --reload
Terminal 2 (Frontend Client):

bash
python -m http.server 8080
Navigate to http://localhost:8080 in your browser, toggle the "🤖 Ask AI (RAG)" switch, and start interrogating the Hacker News hive mind.

🧪 Engineering Deep Dive
Prompt Engineering & Hallucination Mitigation
To prevent the LLM from hallucinating, the backend injects a strict system prompt:

"Answer the user's question using ONLY the context below. Cite your sources using [1], [2] etc. If the context doesn't contain the answer, say 'I don't know based on these stories.'"

Data Pipeline Optimization
The ingest.py script implements asynchronous fetching with rate-limiting (time.sleep) to respect the Hacker News Firebase API's limits, ensuring reliable data ingestion without IP bans.

🗺️ Future Roadmap (v2.0)
□ Hybrid Search Implementation: Combine FAISS with BM25 (via rank_bm25) using Reciprocal Rank Fusion (RRF) to improve exact-match queries.
□ Comment Ingestion: Recursively fetch the top 5 comments per thread to provide the LLM with richer, debate-oriented context.
□ RAG Evaluation Pipeline: Implement a local evaluation harness to track hit@k retrieval accuracy and LLM faithfulness.
□ Cloud-Ready Deployment: Dockerize the application and swap Ollama for a cloud endpoint (e.g., Groq/DeepSeek) to support public hosting on Render/Fly.io.
