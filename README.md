# 🔥 HN-Sage: Semantic Search & RAG over Hacker News

**HN-Sage** is an end-to-end Retrieval-Augmented Generation (RAG) pipeline and web application. It upgrades the traditional Hacker News reading experience by allowing users to ask natural language questions about live tech discussions. 

Instead of simple keyword matching, HN-Sage uses dense vector embeddings to find semantically relevant stories and uses a local Large Language Model to synthesize a cited, hallucination-free answer.

## 🏗️ System Architecture

The application is split into a vanilla JavaScript frontend and a Python FastAPI backend. 

```text
┌────────────────────────┐
│   Frontend (Vanilla)   │
│   HTML / CSS / JS      │
└───────────┬────────────┘
            │ HTTP REST (JSON)
┌───────────▼────────────┐
│   Backend (FastAPI)    │
│   `server.py`          │
└─────┬────────────┬─────┘
      │            │
┌─────▼─────┐  ┌───▼──────────────┐
│  FAISS    │  │  Ollama (Local)  │
│  Vector DB│  │  Llama 3.2       │
└─────▲─────┘  └──────────────────┘
      │
┌─────┴─────┐
│ Embedding │ 
│ Model     │ 
│ (MiniLM)  │
└───────────┘
✨ Features
Hybrid UI: Seamlessly toggle between traditional keyword filtering and AI-powered semantic search on the same page.

Vector Retrieval: Uses sentence-transformers to embed the meaning of Hacker News stories, avoiding the pitfalls of exact-match keyword searches.

100% Local AI: Powered by Ollama (Llama 3.2). No API keys are required, no data leaves the machine, and inference costs are zero.

Source Citations: The LLM is prompted to cite its sources (e.g., [1], [2]) which are rendered below the AI response as clickable Hacker News links.

Modern UI/UX: Responsive, light/dark mode supported, keyboard shortcuts (/ to search, R to reload), and skeleton loading states.

🧠 How the RAG Pipeline Works
Ingestion (ingest.py): Fetches top stories from the official Hacker News API. The title and text of each story are concatenated and passed through the all-MiniLM-L6-v2 embedding model.

Indexing: The resulting dense vectors are normalized and stored in a local FAISS (Facebook AI Similarity Search) index. Metadata (title, author, score) is saved to SQLite.

Retrieval: When a user submits a query, it is embedded into the same vector space. FAISS performs an inner-product (cosine similarity) search to retrieve the top 5 most relevant stories.

Generation: The retrieved stories are injected into a strict prompt context window and sent to Llama 3.2 via Ollama. The model synthesizes an answer using only the provided context.

🛠️ Tech Stack
Machine Learning: Hugging Face (sentence-transformers), FAISS (Vector Database)

LLM Engine: Ollama (Llama 3.2)

Backend: Python, FastAPI, Uvicorn, SQLite

Frontend: Vanilla HTML5, CSS3, JavaScript (ES6+)

💻 Local Setup & Installation
Prerequisites
Python 3.9+

Ollama installed on your machine.

1. Install the AI Model
Open a terminal and pull the Llama model:

bash
ollama run llama3.2
(You can type /bye to exit the chat once the 2GB download finishes).

2. Set Up the Python Environment
Clone this repository and navigate into the folder:

bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
3. Build the Vector Database
Run the ingestion script to fetch the latest Hacker News stories and embed them. This generates the local hn.db, hn.faiss, and ids.json files.

bash
python ingest.py
4. Start the Application
You need two terminals running simultaneously.

Terminal 1 (The AI Backend):

bash
uvicorn server:app --reload
Terminal 2 (The Frontend):

bash
python -m http.server 8080
5. Open the App
Go to http://localhost:8080 in your web browser. Click the "🤖 Ask AI (RAG)" button and start asking questions!
