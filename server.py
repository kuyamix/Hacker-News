# server.py
import json, os, sqlite3, time, logging
from contextlib import asynccontextmanager

import httpx
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
DB_PATH      = os.getenv("HN_DB", "hn.db")
INDEX_PATH   = os.getenv("HN_INDEX", "hn.faiss")
IDS_PATH     = os.getenv("HN_IDS", "ids.json")

log = logging.getLogger("hn_rag")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

# ---------- Lifespan (load once, not at import) ----------
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading embedding model…")
    state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Loading FAISS index…")
    state["index"] = faiss.read_index(INDEX_PATH)
    with open(IDS_PATH) as f:
        state["ids"] = json.load(f)
    log.info(f"Index has {state['index'].ntotal} vectors, {len(state['ids'])} id slots.")
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)


# ---------- DB helpers (per-request connection) ----------
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_story_rows(conn, story_ids):
    """Return rows for the given story ids, keyed by id."""
    if not story_ids:
        return {}
    qs = ",".join("?" * len(story_ids))
    rows = conn.execute(
        f"""SELECT id, title, url, by, score, descendants, time, text, comments_json
            FROM stories WHERE id IN ({qs})""",
        story_ids,
    ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def to_hit(row, similarity):
    comments = json.loads(row.get("comments_json") or "[]")
    return {
        "id": row["id"], "title": row["title"], "url": row["url"],
        "by": row["by"], "score": row["score"],
        "descendants": row.get("descendants", 0),
        "time": row.get("time", 0),
        "text": row.get("text") or "",
        "comments": comments,
        "similarity": round(float(similarity), 3),
    }


# ---------- Retrieval ----------
def _vector_search(query, k):
    model = state["model"]
    index = state["index"]
    ids   = state["ids"]
    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    k = min(k, index.ntotal)
    scores, idxs = index.search(vec, k)
    # Dedupe by story id — keep the best score seen. This handles the
    # "append-only FAISS may contain stale duplicates" case.
    best = {}
    for score, i in zip(scores[0], idxs[0]):
        if i < 0 or i >= len(ids):
            continue
        sid = ids[int(i)]
        if sid not in best or score > best[sid]:
            best[sid] = float(score)
    return best  # {story_id: similarity}


def _bm25_scores(query, conn, k=30):
    """Simple in-memory BM25 over title+text+comments for the candidate pool."""
    rows = conn.execute(
        "SELECT id, title, text, comments_json FROM stories").fetchall()
    if not rows:
        return {}
    docs = []
    for r in rows:
        comments = json.loads(r["comments_json"] or "[]")
        comments_text = " ".join(c["text"] for c in comments)
        docs.append((r["id"], f"{r['title']} {r['text'] or ''} {comments_text}"))

    terms = [t.lower() for t in query.split() if t.strip()]
    if not terms:
        return {}

    from math import log
    N = len(docs)
    df = {t: 0 for t in terms}
    tokenized = []
    for _, text in docs:
        toks = text.lower().split()
        tokenized.append(toks)
        for t in terms:
            if any(t in tok for tok in toks):
                df[t] += 1
    avgdl = sum(len(t) for t in tokenized) / max(N, 1) or 1.0
    k1, b = 1.5, 0.75

    scores = {}
    for (sid, _), toks in zip(docs, tokenized):
        s = 0.0
        for t in terms:
            tf = sum(1 for tok in toks if t in tok)
            if tf == 0:
                continue
            idf = log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            denom = tf + k1 * (1 - b + b * len(toks) / avgdl)
            s += idf * tf * (k1 + 1) / denom
        if s > 0:
            scores[sid] = s
    # Return only the top-k ids (scores normalized by max for RRF input).
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return dict(top)


def hybrid_search(query, k=5, pool=30):
    """Vector + BM25, fused with Reciprocal Rank Fusion."""
    with db_conn() as conn:
        vec = _vector_search(query, pool)
        bm  = _bm25_scores(query, conn, pool)

        # RRF: score = sum over rankers of 1/(60 + rank)
        ranks = {}
        for r, sid in enumerate(sorted(vec, key=vec.get, reverse=True)):
            ranks[sid] = ranks.get(sid, 0.0) + 1.0 / (60 + r + 1)
        for r, sid in enumerate(sorted(bm, key=bm.get, reverse=True)):
            ranks[sid] = ranks.get(sid, 0.0) + 1.0 / (60 + r + 1)

        top_ids = sorted(ranks, key=ranks.get, reverse=True)[:k]
        rows = fetch_story_rows(conn, top_ids)

        hits = []
        for sid in top_ids:
            if sid not in rows:
                continue
            # Display vector similarity when available, else fused score.
            sim = vec.get(sid, ranks[sid])
            hits.append(to_hit(rows[sid], sim))
        return hits


# ---------- Prompt ----------
def build_prompt(query, hits):
    context_parts = []
    for i, h in enumerate(hits):
        block = [f"[{i+1}] Title: {h['title']}", f"By: {h['by']} · {h['score']} points"]
        if h["text"]:
            block.append(f"Self-text: {h['text'][:800]}")
        if h["comments"]:
            for j, c in enumerate(h["comments"][:3], 1):
                block.append(f"Comment {j} (by {c['by']}): {c['text'][:400]}")
        context_parts.append("\n".join(block))

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an analyst summarizing Hacker News stories.

Rules:
- Use the provided context as your primary evidence.
- Cite every factual claim about the stories with [n] matching source numbers.
- When a claim comes from a comment rather than the title, note it: "commenters note [n]".
- You may add general background knowledge, but mark it clearly as "(general knowledge)".
- If the context doesn't answer the question, say so explicitly.
- Be concise (under 200 words unless the question requires more).
- Quote a short phrase from the source for any key claim, not just the citation number.

Context:
{context}

Question: {query}

Answer:"""


# ---------- Endpoints ----------
@app.get("/health")
def health():
    try:
        with db_conn() as conn:
            n = conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0]
        return {
            "ok": True,
            "index_vectors": state["index"].ntotal,
            "story_rows": n,
            "model_loaded": "model" in state,
            "llm_model": OLLAMA_MODEL,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/search")
def search(q: str, k: int = Query(5, ge=1, le=20)):
    return hybrid_search(q, k=k)


@app.get("/ask")
def ask(q: str, k: int = Query(5, ge=1, le=10)):
    hits = hybrid_search(q, k=k)
    if not hits:
        return {"answer": "No relevant stories found.", "sources": []}

    prompt = build_prompt(q, hits)
    t0 = time.perf_counter()

    try:
        with httpx.Client(timeout=90) as client:
            r = client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"num_predict": 512, "temperature": 0.3},
                },
            )
            r.raise_for_status()
            answer = r.json()["message"]["content"]
    except Exception as e:
        answer = (
            f"Could not reach the local LLM. Is Ollama running? "
            f"Try `ollama run {OLLAMA_MODEL}` in a terminal. Error: {e}"
        )

    log.info(f"/ask q_len={len(q)} hits={len(hits)} "
             f"elapsed={time.perf_counter()-t0:.2f}s")
    return {"answer": answer, "sources": hits}


@app.get("/ask/stream")
def ask_stream(q: str, k: int = Query(5, ge=1, le=10)):
    """Server-Sent-Events style token stream. Emits one JSON object per line.
    First event: {"type":"sources","data":[...]}
    Then:        {"type":"token","data":"..."}
    Final:       {"type":"done"}
    """
    hits = hybrid_search(q, k=k)
    if not hits:
        async def empty():
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            yield json.dumps({"type": "token", "data": "No relevant stories found."}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        return StreamingResponse(empty(), media_type="application/x-ndjson")

    prompt = build_prompt(q, hits)

    async def gen():
        yield json.dumps({"type": "sources", "data": hits}) + "\n"
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                        "options": {"num_predict": 512, "temperature": 0.3},
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
                            yield json.dumps({"type": "token", "data": chunk}) + "\n"
                        if obj.get("done"):
                            break
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")