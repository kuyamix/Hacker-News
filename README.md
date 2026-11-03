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
