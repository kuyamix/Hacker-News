# HN-Sage
### Semantic Search & Local RAG for Hacker News

> "Stop keyword searching. Start asking questions."

HN-Sage is a lightweight, fully-local Retrieval-Augmented Generation (RAG) pipeline built on top of the Hacker News API. It replaces traditional keyword filtering with vector similarity search and uses a local Llama 3.2 model to synthesize answers from live tech discussions.

<br>

## ⚡ The Stack

```text
[ Frontend ]  Vanilla JS + CSS (Zero dependencies, Dark Mode)
     │
     ▼
[ Backend  ]  Python 3.9+ / FastAPI / Uvicorn
     │
     ├──▶ [ Vector DB ]  FAISS (IndexFlatIP)
     ├──▶ [ Embeddings]  all-MiniLM-L6-v2 (384d)
     └──▶ [ LLM Engine]  Ollama (Llama 3.2)

🧠 How it works
Ingest: Scrapes the top stories from the HN Firebase API.

Embed: Converts titles and text into dense vectors.

Index: Stores vectors in FAISS for sub-millisecond retrieval.

Retrieve: Embeds your query, finds the top-5 most similar stories.

Generate: Injects the retrieved context into a prompt and asks Llama 3.2 to answer with citations.


🚀 Run it locally
1. Setup the local AI

bash
ollama run llama3.2
2. Setup the project

bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
3. Build the vector index

bash
python ingest.py
4. Fire it up (Two terminals)

bash
# Terminal 1: AI Backend
uvicorn server:app --reload

# Terminal 2: Frontend
python -m http.server 8080
Open http://localhost:8080. Toggle "🤖 Ask AI" and start interrogating the hive mind.


📂 Project Structure
text
├── ingest.py          # Data pipeline: Fetch HN API -> Embed -> Save to FAISS/SQLite
├── server.py          # FastAPI backend: /search and /ask endpoints
├── index.html         # Hybrid UI: Keyword filter + AI RAG interface
├── hn.db              # SQLite (Story metadata)
├── hn.faiss           # FAISS (Vector embeddings)
└── ids.json           # FAISS index to SQLite ID mapping

🛠️ Technical Decisions
Why FAISS over ChromaDB? FAISS is significantly faster for in-memory vector search and requires zero setup overhead for local prototyping.

Why Ollama? Cloud APIs (OpenAI, Anthropic) cost money and send your data to third parties. Ollama runs Llama 3.2 locally on CPU/Mac Silicon, making this pipeline 100% free and private.

Why Vanilla JS? To prove that you don't need a 500MB node_modules folder to build a fast, responsive, and beautiful UI.


🗺️ What's Next?
□ Hybrid Search (Combine BM25 + Vector search)
□ Comment Summarization (Fetch top comments for richer context)
□ Deploy to Cloud (Swap Ollama for Groq API + Deploy FastAPI to Render)
□ Eval Harness (Track retrieval hit@5 with an LLM-as-a-judge)
