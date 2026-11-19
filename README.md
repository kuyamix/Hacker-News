# HN-Sage: Local-First RAG for Hacker News

A privacy-focused, provider-agnostic Retrieval-Augmented Generation (RAG) application that transforms Hacker News discussions into a searchable knowledge base powered by Large Language Models (LLMs).
---

## 📖 Overview

**HN-Sage** is a local-first RAG application that collects live discussions from Hacker News, processes and indexes their content into a local vector database, and allows users to ask questions using natural language.

Instead of relying entirely on an LLM's pre-trained knowledge, HN-Sage retrieves relevant information from indexed Hacker News stories and comments before generating an answer. Responses include source citations, allowing users to trace generated information back to the original discussions.

The application addresses two major challenges in modern AI systems:

1. **The Hallucination Problem** – LLMs can generate inaccurate or outdated information. HN-Sage uses retrieved source material to ground responses in actual Hacker News discussions.
2. **The Redundancy Problem** – Traditional retrieval methods can return multiple similar passages. HN-Sage uses Maximal Marginal Relevance (MMR) to diversify retrieved results and provide broader context.

### Key Features

- **Local-First Architecture:** Run LLM inference locally using Ollama.
- **Semantic Search:** Find relevant discussions based on meaning rather than exact keywords.
- **Hybrid Retrieval:** Combine BM25 keyword search with vector similarity search.
- **MMR Diversification:** Reduce repetitive search results and improve contextual diversity.
- **Source Citations:** Connect generated answers to their original Hacker News stories and comments.
- **Streaming Responses:** Display generated answers as they are produced.
- **Provider-Agnostic Inference:** Switch between Ollama, OpenAI, and Groq through environment variables.

---

## 🏗️ Architecture & Emerging Technologies

HN-Sage integrates four emerging technologies to create a flexible and efficient AI-powered information retrieval system.

### 1. Local LLM Inference (Ollama)

The application uses **Ollama** to run the `llama3.2` language model locally.

This provides:
- Improved data privacy by keeping inference local.
- No per-request cloud inference charges.
- Reduced dependency on external AI services.
- The ability to operate without sending retrieved discussion content to a cloud provider.

### 2. Vector Search (FAISS)

HN-Sage uses **Facebook AI Similarity Search (FAISS)** alongside `sentence-transformers` to convert text passages into numerical embeddings.

The default embedding model, `all-MiniLM-L6-v2`, produces 384-dimensional vectors.

These embeddings allow the system to perform semantic searches, identifying passages that are conceptually related to a query even when they do not share the same keywords.

### 3. Hybrid Retrieval (BM25 + Vector Search)

The application combines two retrieval approaches:

- **BM25:** Finds relevant passages using keyword matching and lexical relevance.
- **Vector Search:** Identifies semantically similar passages using embedding similarity.

Their results are combined using **Reciprocal Rank Fusion (RRF)**, producing a unified ranking that benefits from both retrieval methods.

### 4. Provider-Agnostic LLM Abstraction

The inference layer is separated from the retrieval pipeline through a provider abstraction.

This allows HN-Sage to switch between different LLM providers without modifying the underlying retrieval logic.

Supported providers include:
- Ollama (Local)
- OpenAI
- Groq (OpenAI-compatible API)

Provider selection and configuration are controlled through environment variables.

### System Architecture

```text
┌──────────────────────────────┐
│       Frontend (HTML/JS)     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│         FastAPI Server       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Hybrid Retrieval       │
│                              │
│       FAISS + BM25/SQLite    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Maximal Marginal          │
│    Relevance (MMR)            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       LLM Abstraction        │
│                              │
│  Ollama / OpenAI / Groq      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Streamed Response       │
│      with Source Citations   │
└──────────────────────────────┘
```

---

## 🧠 Engineering Deep Dives

### 1. Passage-Level Chunking (`ingest.py`)

Rather than embedding an entire Hacker News discussion as one large text block, HN-Sage uses passage-level chunking.

Each story is divided into smaller, independently searchable passages:

| Chunk | Content |
|---|---|
| Chunk 0 | Story title and self-text |
| Chunk 1 | First top-level comment |
| Chunk 2 | Second top-level comment |
| Chunk 3 | Third top-level comment |
| Chunk 4 | Fourth top-level comment |
| Chunk 5 | Fifth top-level comment |

This approach allows the retrieval engine to identify the specific comment that contains relevant information instead of returning an entire discussion.

**Benefit:** Smaller, more focused passages improve retrieval precision and allow the LLM to receive more relevant context.

### 2. Maximal Marginal Relevance (MMR) (`server.py`)

To address repetitive retrieval results, HN-Sage implements **Maximal Marginal Relevance (MMR)** with a default lambda value of `0.5`.

MMR considers two factors when selecting passages:

- **Relevance:** How closely a passage matches the user's query.
- **Diversity:** How different a passage is from previously selected passages.

The algorithm balances these factors to select passages that are both relevant and diverse.

The MMR selection formula is:

\[
MMR = \lambda \cdot Sim(D,Q)
-
(1-\lambda)\cdot
\max_{D_j \in S} Sim(D,D_j)
\]

Where:

- \(D\) = Candidate passage.
- \(Q\) = User query.
- \(S\) = Previously selected passages.
- \(Sim(D,Q)\) = Relevance to the query.
- \(Sim(D,D_j)\) = Similarity to an already selected passage.
- \(\lambda\) = Relevance-diversity balance.

With `lambda=0.5`, the system gives equal weight to relevance and diversity.

**Benefit:** The LLM receives a broader range of perspectives rather than multiple nearly identical comments.

### 3. LLM Abstraction Layer

The backend separates language model inference from information retrieval.

The retrieval engine does not depend on a specific LLM provider. Instead, it communicates through a common inference interface.

This makes it possible to change the underlying model without rewriting the retrieval pipeline.

For example:

| Provider | Inference Method |
|---|---|
| Ollama | Local model inference |
| OpenAI | Cloud-based API |
| Groq | Cloud-based OpenAI-compatible API |

**Benefit:** The application remains flexible, easier to maintain, and adaptable to different deployment environments.

---

## 🚀 Installation & Setup

### Prerequisites

Before running HN-Sage, ensure that the following are installed:

- Python 3.9 or later
- [Ollama](https://ollama.com/) (for local LLM inference)
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd HN-Sage
```

Alternatively, download and extract the project files.

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

It is recommended to use a virtual environment:

```bash
python -m venv venv
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

Then install the dependencies.

### 3. Ingest Hacker News Data

Run the ingestion script:

```bash
python ingest.py
```

The script retrieves Hacker News stories and comments through the Hacker News API, processes the content, generates embeddings, and creates the local index.

The following files are generated:

| File | Purpose |
|---|---|
| `hn.db` | Stores indexed story and comment data |
| `hn.faiss` | Stores vector embeddings for similarity search |
| `ids.json` | Maps vector identifiers to their corresponding records |

These files are generated locally and are excluded from Git through `.gitignore`.

**Note:** Internet access is required during ingestion to retrieve Hacker News data.

### 4. Start the Local LLM

Install and start Ollama, then download the default model:

```bash
ollama pull llama3.2
```

Start the model:

```bash
ollama run llama3.2
```

Keep Ollama running while using the application.

### 5. Start the Backend Server

Open a separate terminal in the project directory and execute:

```bash
uvicorn server:app --reload
```

The FastAPI backend will start with automatic reloading enabled.

### 6. Open the Frontend

Open `index.html` in your browser.

Alternatively, serve the frontend using Python's built-in HTTP server:

```bash
python -m http.server 3000
```

Then navigate to:

```text
http://localhost:3000
```

---

## ⚙️ Configuration & Ablation Testing

HN-Sage supports different retrieval configurations through environment variables.

These configurations allow users to compare retrieval strategies and observe the effects of MMR on search quality.

The following examples use Windows PowerShell.

### 1. Default Configuration

Uses hybrid retrieval with MMR enabled.

```powershell
uvicorn server:app
```

### 2. BM25-Only Retrieval

Disables semantic vector retrieval and uses keyword-based search.

```powershell
$env:RETRIEVAL_MODE="bm25"
uvicorn server:app
```

### 3. Vector-Only Retrieval

Uses semantic similarity search without BM25 keyword retrieval.

```powershell
$env:RETRIEVAL_MODE="vector"
uvicorn server:app
```

### 4. Disable MMR Diversification

Setting `MMR_LAMBDA` to `1.0` prioritizes relevance over diversity, effectively removing the redundancy penalty.

```powershell
$env:MMR_LAMBDA="1.0"
uvicorn server:app
```

### Configuration Comparison

| Configuration | Retrieval Method | Expected Behavior |
|---|---|---|
| Default | Hybrid + MMR | Balanced relevance and diversity |
| BM25 | Keyword-based | Prioritizes exact keyword matches |
| Vector | Semantic | Prioritizes conceptual similarity |
| Lambda = 1.0 | Hybrid without diversity penalty | May return repetitive passages |

These configurations provide a foundation for evaluating the application's retrieval performance.

---

## ☁️ Optional: Cloud LLM Inference

Although HN-Sage defaults to local Ollama inference, it can also use cloud-based providers such as OpenAI or Groq.

To configure cloud inference, set the appropriate environment variables.

**Example: Groq Configuration**

```powershell
$env:LLM_PROVIDER="openai"
$env:LLM_API_KEY="your-api-key-here"
$env:LLM_BASE_URL="https://api.groq.com/openai/v1"
$env:LLM_MODEL="llama-3.1-70b-versatile"

uvicorn server:app --reload
```

**Important:** The example uses the OpenAI-compatible API interface with Groq's endpoint. Ensure that the model identifier is supported by your provider.

For OpenAI, configure the provider's API key, base URL, and model according to the implementation in the backend.

Keep API keys private. Never commit credentials or `.env` files to a public repository.

---

## 📁 Project Structure

The project is organized around data ingestion, retrieval, inference, and frontend presentation.

```text
HN-Sage/
│
├── ingest.py              # Hacker News data ingestion
├── server.py              # FastAPI backend and RAG pipeline
├── index.html             # Frontend interface
├── requirements.txt       # Python dependencies
├── .gitignore             # Excluded files and directories
│
├── hn.db                  # Local SQLite database
├── hn.faiss               # FAISS vector index
├── ids.json               # Vector-to-record mapping
│
└── README.md              # Project documentation
```

*Note: The structure above represents the main project files. Actual directories may vary depending on the implementation.*

---

## 🔮 Future Improvements

HN-Sage provides a foundation for local-first AI-powered information retrieval. Several improvements could further expand its capabilities.

### 1. Agentic Query Planning

Introduce an LLM-based query planning mechanism that transforms complex user questions into multiple smaller sub-queries.

This could improve retrieval coverage by identifying different aspects of a question before searching the knowledge base.

### 2. Expanded Hacker News Coverage

Expand the `FEEDS` configuration in `ingest.py` to include additional Hacker News categories, such as:

- Best Stories
- Ask HN
- Show HN
- New Stories

This would provide a larger and more diverse collection of discussions.

### 3. Improved Embedding Models

Experiment with newer embedding models, such as `nomic-embed-text`, to evaluate improvements in semantic retrieval quality.

Any embedding model change would require regenerating the existing vector index to maintain embedding compatibility.

### 4. Advanced Retrieval Evaluation

Introduce a formal evaluation framework to measure:

- Retrieval precision and recall.
- Context relevance.
- Answer faithfulness.
- Citation accuracy.
- Retrieval latency.

This would make it easier to compare different retrieval configurations objectively.

### 5. Persistent Conversation History

Add conversation history so users can ask follow-up questions while maintaining context from previous interactions.

---

## 🔒 Privacy & Data Handling

HN-Sage is designed around a local-first architecture.

- Hacker News content is downloaded and indexed locally.
- SQLite and FAISS store the local knowledge base.
- Ollama performs inference locally by default.
- Cloud providers are optional and require explicit configuration.

**Privacy consideration:** Using OpenAI or Groq sends prompts and retrieved context to the configured external provider. Therefore, the privacy guarantees of local-only inference do not apply when cloud inference is enabled.

---

## 🎓 Project Summary

HN-Sage demonstrates how emerging AI technologies can be integrated into a practical information retrieval application.

By combining Retrieval-Augmented Generation, vector databases, hybrid search, MMR diversification, and provider-independent LLM inference, the project demonstrates an approach to building AI applications that are more grounded, flexible, and privacy-conscious.

The project highlights the importance of combining information retrieval with language generation rather than relying solely on an LLM's pre-trained knowledge.

**HN-Sage: Turning Hacker News discussions into a searchable, AI-powered knowledge base.**
