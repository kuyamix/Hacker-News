# server.py
"""
HN-Sage FastAPI server.

Phase 1 changes:
  - Schema migrations run at startup (db.migrate).
  - FAISS-row → (story_id, chunk_idx) mapping comes from the chunk_vectors
    table instead of ids.json. ids.json is no longer read.
  - BM25 is now FTS5-backed (one SQL query) instead of the hand-rolled
    tokenizer scan. Sign-flipped so higher-is-better, matching the rest of
    the pipeline.
  - /health returns the fields the frontend expects (story_rows, llm_model).

Phase 0 additions folded in:
  - GET /ask: non-streaming JSON answer + sources (used by test.html and
    any eval script).

Unchanged from v1:
  - _mmr_rerank (the "chunk 0 vector" issue is deliberately preserved for
    phase 2, so we can attribute any regression to the data-layer change).
  - build_prompt, hybrid_search, /search, /ask/stream shapes.

Env vars:
  OLLAMA_MODEL       (default llama3.2)
  OLLAMA_URL         (default http://localhost:11434)
  HN_DB              (default hn.db)
  HN_INDEX           (default hn.faiss)
  EMBED_MODEL        (default all-MiniLM-L6-v2)
  RETRIEVAL_MODE     hybrid | vector | bm25   (default hybrid)
  MMR_LAMBDA         (default 0.5)
"""

import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Any

import faiss
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer

from db import get_db, migrate

# ---------- Config ----------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
DB_PATH      = os.getenv("HN_DB", "hn.db")
INDEX_PATH   = os.getenv("HN_INDEX", "hn.faiss")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")   # hybrid | vector | bm25
MMR_LAMBDA     = float(os.getenv("MMR_LAMBDA", "0.5"))

log = logging.getLogger("hn_rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
state: dict[str, Any] = {}

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Ensuring schema…")
    migrate()

    log.info("Loading embedding model (%s)…", EMBED_MODEL)
    state["model"] = SentenceTransformer(EMBED_MODEL)

    log.info("Loading FAISS index…")
    state["index"] = faiss.read_index(INDEX_PATH)
    log.info("Materializing vectors for MMR…")
    state["all_vectors"] = state["index"].reconstruct_n(0, state["index"].ntotal)

    log.info("Loading chunk_vectors…")
    with get_db() as conn:
        rows = conn.execute(
            "SELECT faiss_row, story_id, chunk_idx FROM chunk_vectors ORDER BY faiss_row"
        ).fetchall()
    state["row_to_pair"] = {
        int(r["faiss_row"]): (int(r["story_id"]), int(r["chunk_idx"])) for r in rows
    }

    if state["index"].ntotal != len(state["row_to_pair"]):
        log.warning(
            "FAISS ntotal (%d) != chunk_vectors rows (%d). "
            "Run migrate_v1_to_v2.py or re-ingest before relying on retrieval.",
            state["index"].ntotal, len(state["row_to_pair"]),
        )

    # story_id -> faiss_row of its FIRST chunk. Used by MMR.
    # NOTE: phase 2 will change this to "best-matching chunk per query".
    id_to_rowidx: dict[int, int] = {}
    for faiss_row, (sid, _cidx) in state["row_to_pair"].items():
        id_to_rowidx.setdefault(sid, faiss_row)
    state["id_to_rowidx"] = id_to_rowidx

    log.info(
        "Ready. %d vectors, %d chunks, retrieval_mode=%s, mmr_lambda=%.2f",
        state["index"].ntotal, len(state["row_to_pair"]),
        RETRIEVAL_MODE, MMR_LAMBDA,
    )
    try:
        yield
    finally:
        state.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB helpers ----------
def db_conn():
    return get_db()

def fetch_story_rows(conn, story_ids):
    if not story_ids:
        return {}
    qs = ",".join("?" * len(story_ids))
    rows = conn.execute(
        f"SELECT * FROM stories WHERE id IN ({qs})", story_ids
    ).fetchall()
    return {r["id"]: dict(r) for r in rows}

def to_hit(row, similarity, matched_chunk_idx, matched_doc):
    return {
        "id": row["id"],
        "title": row["title"],
        "url": row["url"],
        "by": row["by"],
        "score": row["score"],
        "descendants": row.get("descendants", 0),
        "time": row.get("time", 0),
        "text": row.get("text") or "",
        "comments": json.loads(row.get("comments_json") or "[]"),
        "similarity": round(float(similarity), 3),
        "matched_chunk_idx": matched_chunk_idx,
        "matched_doc": matched_doc,
    }

# ---------- Retrieval ----------
def _vector_search(query: str, k: int) -> dict[int, tuple[float, int, str]]:
    """
    Dense retrieval. Returns story_id -> (score, chunk_idx, doc) of the
    story's best-matching chunk.
    """
    model, index = state["model"], state["index"]
    row_to_pair = state["row_to_pair"]

    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    k = min(k, index.ntotal)
    scores, idxs = index.search(vec, k)

    best: dict[int, tuple[float, int, str]] = {}
    with db_conn() as conn:
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            pair = row_to_pair.get(int(i))
            if not pair:
                continue
            story_id, chunk_idx = pair
            c_row = conn.execute(
                "SELECT doc FROM chunks WHERE story_id=? AND chunk_idx=?",
                (story_id, chunk_idx),
            ).fetchone()
            doc = c_row["doc"] if c_row else ""
            if story_id not in best or score > best[story_id][0]:
                best[story_id] = (float(score), chunk_idx, doc)
    return best

def _bm25_scores(query: str, conn, k: int = 30) -> dict[int, tuple[float, int, str]]:
    """
    FTS5-backed BM25. bm25() is negative-better, so we flip the sign so
    higher-is-better (matching the rest of the pipeline and RRF ranking).
    """
    safe = " ".join(t for t in query.split() if t.isalnum() or "-" in t)
    if not safe:
        return {}
    try:
        rows = conn.execute(
            """
            SELECT chunks.story_id AS story_id,
                   bm25(chunks_fts) AS bm25,
                   chunks.chunk_idx AS chunk_idx,
                   chunks.doc AS doc
            FROM chunks_fts
            JOIN chunks ON chunks.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25
            LIMIT ?
            """,
            (safe, k),
        ).fetchall()
    except sqlite3.OperationalError as e:
        log.warning("FTS query failed (%s); returning empty BM25", e)
        return {}

    best: dict[int, tuple[float, int, str]] = {}
    for r in rows:
        s = -float(r["bm25"])   # flip sign
        sid = int(r["story_id"])
        if sid not in best or s > best[sid][0]:
            best[sid] = (s, int(r["chunk_idx"]), r["doc"])
    return best

def _mmr_rerank(query_vec, candidate_ids, relevance, k, lambda_=MMR_LAMBDA):
    if not candidate_ids:
        return []
    vecs = state.get("all_vectors")
    rowidx = state.get("id_to_rowidx", {})

    if vecs is None or not rowidx:
        return sorted(candidate_ids, key=lambda s: relevance[s][0], reverse=True)[:k]

    max_rel = max((relevance[s][0] for s in candidate_ids), default=1.0) or 1.0
    selected, remaining = [], list(candidate_ids)

    while remaining and len(selected) < k:
        best_id, best_score = None, -1e9
        for sid in remaining:
            rel = relevance[sid][0] / max_rel
            v = vecs[rowidx[sid]]
            div = max(
                (float(np.dot(v, vecs[rowidx[s]])) for s in selected), default=0.0
            ) if selected else 0.0
            mmr = lambda_ * rel - (1 - lambda_) * div
            if mmr > best_score:
                best_score, best_id = mmr, sid
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    return selected

def hybrid_search(query: str, k: int = 5, pool: int = 30) -> list[dict]:
    model = state["model"]
    qvec = model.encode([query], normalize_embeddings=True).astype("float32")[0]

    with db_conn() as conn:
        vec = {} if RETRIEVAL_MODE == "bm25" else _vector_search(query, pool)
        bm  = {} if RETRIEVAL_MODE == "vector" else _bm25_scores(query, conn, pool)

        ranks: dict[int, float] = {}
        for r, sid in enumerate(sorted(vec, key=lambda s: vec[s][0], reverse=True)):
            ranks[sid] = ranks.get(sid, 0.0) + 1.0 / (60 + r + 1)
        for r, sid in enumerate(sorted(bm, key=lambda s: bm[s][0], reverse=True)):
            ranks[sid] = ranks.get(sid, 0.0) + 1.0 / (60 + r + 1)

        candidate_ids = sorted(ranks, key=ranks.get, reverse=True)[:pool]

        relevance: dict[int, tuple[float, int, str]] = {}
        for sid in candidate_ids:
            relevance[sid] = vec[sid] if sid in vec else bm[sid]

        top_ids = _mmr_rerank(qvec, candidate_ids, relevance, k)
        rows = fetch_story_rows(conn, top_ids)

        hits = []
        for sid in top_ids:
            if sid not in rows:
                continue
            score, c_idx, doc = relevance[sid]
            hits.append(to_hit(rows[sid], score, c_idx, doc))
        return hits

def build_prompt(query: str, hits: list[dict]) -> str:
    context_parts = []
    for i, h in enumerate(hits):
        block = [
            f"[{i+1}] Title: {h['title']}",
            f"By: {h['by']} · {h['score']} points",
        ]
        if h["matched_doc"]:
            block.append(f"Matched Passage:\n{h['matched_doc'][:1500]}")
        context_parts.append("\n".join(block))
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an analyst summarizing Hacker News stories.
Use the provided context as your primary evidence.
Cite every factual claim with [n] matching source numbers.
Draw on as many distinct sources as possible.

Context:
{context}

Question: {query}

Answer:"""

# ---------- Ollama ----------
async def _ollama_chat_stream(prompt: str, num_predict: int = 512, temperature: float = 0.3):
    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": {"num_predict": num_predict, "temperature": temperature},
            },
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = obj.get("message", {}).get("content", "")
                if chunk:
                    yield chunk

async def _ollama_chat_once(prompt: str, num_predict: int = 512, temperature: float = 0.3) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": num_predict, "temperature": temperature},
            },
        )
        r.raise_for_status()
        body = r.json()
        return body.get("message", {}).get("content", "")

# ---------- Routes ----------
@app.get("/health")
def health():
    with db_conn() as conn:
        story_rows = conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0]
        chunk_rows = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {
        "ok": True,
        "story_rows": story_rows,
        "chunk_rows": chunk_rows,
        "vectors": state["index"].ntotal,
        "llm_model": OLLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "retrieval_mode": RETRIEVAL_MODE,
        "mmr_lambda": MMR_LAMBDA,
    }

@app.get("/search")
def search(q: str, k: int = Query(10, ge=1, le=30)):
    return hybrid_search(q, k=k)

@app.get("/ask")
async def ask(q: str, k: int = Query(8, ge=1, le=30)):
    """Non-streaming answer + sources. Used by test.html and eval scripts."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty.")
    t0 = time.perf_counter()
    hits = hybrid_search(q, k=k)
    prompt = build_prompt(q, hits)
    try:
        answer = await _ollama_chat_once(prompt)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}") from e
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "answer": answer,
        "sources": hits,
        "latency_ms": latency_ms,
        "model": OLLAMA_MODEL,
    }

@app.get("/ask/stream")
async def ask_stream(q: str, k: int = Query(8, ge=1, le=30)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must be non-empty.")
    hits = hybrid_search(q, k=k)
    prompt = build_prompt(q, hits)

    async def gen():
        yield json.dumps({"type": "sources", "data": hits}) + "\n"
        try:
            async for chunk in _ollama_chat_stream(prompt):
                yield json.dumps({"type": "token", "data": chunk}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")