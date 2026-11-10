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

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
DB_PATH      = os.getenv("HN_DB", "hn.db")
INDEX_PATH   = os.getenv("HN_INDEX", "hn.faiss")
IDS_PATH     = os.getenv("HN_IDS", "ids.json")

# ABLATION FLAGS
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid") # 'hybrid', 'vector', 'bm25'
MMR_LAMBDA     = float(os.getenv("MMR_LAMBDA", "0.5"))

log = logging.getLogger("hn_rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading embedding model…")
    state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
    
    log.info("Loading FAISS index…")
    state["index"] = faiss.read_index(INDEX_PATH)
    with open(IDS_PATH) as f: 
        state["ids"] = json.load(f)
        
    log.info("Materializing vectors for MMR…")
    state["all_vectors"] = state["index"].reconstruct_n(0, state["index"].ntotal)
    
    # FIXED: Handle the new [story_id, chunk_idx] array format
    id_to_rowidx = {}
    for row_i, id_pair in enumerate(state["ids"]):
        story_id = id_pair[0] if isinstance(id_pair, list) else id_pair
        id_to_rowidx.setdefault(story_id, row_i)
    state["id_to_rowidx"] = id_to_rowidx
    
    log.info(f"Index ready. {state['index'].ntotal} vectors loaded.")
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_story_rows(conn, story_ids):
    if not story_ids: return {}
    qs = ",".join("?" * len(story_ids))
    rows = conn.execute(f"SELECT * FROM stories WHERE id IN ({qs})", story_ids).fetchall()
    return {r["id"]: dict(r) for r in rows}

def to_hit(row, similarity, matched_chunk_idx, matched_doc):
    return {
        "id": row["id"], "title": row["title"], "url": row["url"],
        "by": row["by"], "score": row["score"], "descendants": row.get("descendants", 0),
        "time": row.get("time", 0), "text": row.get("text") or "",
        "comments": json.loads(row.get("comments_json") or "[]"),
        "similarity": round(float(similarity), 3),
        "matched_chunk_idx": matched_chunk_idx,
        "matched_doc": matched_doc
    }

def _vector_search(query, k):
    model, index, ids = state["model"], state["index"], state["ids"]
    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    k = min(k, index.ntotal)
    scores, idxs = index.search(vec, k)
    
    best = {} # story_id -> (score, chunk_idx, doc)
    with db_conn() as conn:
        for score, i in zip(scores[0], idxs[0]):
            if i < 0 or i >= len(ids): continue
            story_id, chunk_idx = ids[int(i)]
            
            c_row = conn.execute("SELECT doc FROM chunks WHERE story_id=? AND chunk_idx=?", (story_id, chunk_idx)).fetchone()
            doc = c_row["doc"] if c_row else ""
            
            if story_id not in best or score > best[story_id][0]:
                best[story_id] = (float(score), chunk_idx, doc)
    return best

def _bm25_scores(query, conn, k=30):
    rows = conn.execute("SELECT story_id, doc FROM chunks").fetchall()
    if not rows: return {}
    docs = [(r["story_id"], r["doc"]) for r in rows]
    terms = [t.lower() for t in query.split() if t.strip()]
    if not terms: return {}

    from math import log
    N = len(docs)
    df = {t: 0 for t in terms}
    tokenized = []
    for _, text in docs:
        toks = text.lower().split()
        tokenized.append(toks)
        for t in terms:
            if any(t in tok for tok in toks): df[t] += 1
    avgdl = sum(len(t) for t in tokenized) / max(N, 1) or 1.0
    k1, b = 1.5, 0.75

    best = {}
    for (sid, text), toks in zip(docs, tokenized):
        s = 0.0
        for t in terms:
            tf = sum(1 for tok in toks if t in tok)
            if tf == 0: continue
            idf = log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            denom = tf + k1 * (1 - b + b * len(toks) / avgdl)
            s += idf * tf * (k1 + 1) / denom
        
        if s > 0:
            if sid not in best or s > best[sid][0]:
                best[sid] = (s, -1, text)
                
    return dict(sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:k])

def _mmr_rerank(query_vec, candidate_ids, relevance, k, lambda_=MMR_LAMBDA):
    if not candidate_ids: return []
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
            # Compare against already selected vectors
            div = max((float(np.dot(v, vecs[rowidx[s]])) for s in selected), default=0.0) if selected else 0.0
            mmr = lambda_ * rel - (1 - lambda_) * div
            if mmr > best_score: 
                best_score, best_id = mmr, sid
        if best_id is None: break
        selected.append(best_id)
        remaining.remove(best_id)
    return selected

def hybrid_search(query, k=5, pool=30):
    model = state["model"]
    qvec = model.encode([query], normalize_embeddings=True).astype("float32")[0]
    
    with db_conn() as conn:
        vec = {} if RETRIEVAL_MODE == "bm25" else _vector_search(query, pool)
        bm  = {} if RETRIEVAL_MODE == "vector" else _bm25_scores(query, conn, pool)

        ranks = {}
        for r, sid in enumerate(sorted(vec, key=lambda s: vec[s][0], reverse=True)):
            ranks[sid] = ranks.get(sid, 0.0) + 1.0 / (60 + r + 1)
        for r, sid in enumerate(sorted(bm, key=lambda s: bm[s][0], reverse=True)):
            ranks[sid] = ranks.get(sid, 0.0) + 1.0 / (60 + r + 1)

        candidate_ids = sorted(ranks, key=ranks.get, reverse=True)[:pool]
        
        relevance = {}
        for sid in candidate_ids:
            if sid in vec: relevance[sid] = vec[sid]
            else: relevance[sid] = bm[sid]

        top_ids = _mmr_rerank(qvec, candidate_ids, relevance, k)
        rows = fetch_story_rows(conn, top_ids)
        
        hits = []
        for sid in top_ids:
            if sid not in rows: continue
            score, c_idx, doc = relevance[sid]
            hits.append(to_hit(rows[sid], score, c_idx, doc))
        return hits

def build_prompt(query, hits):
    context_parts = []
    for i, h in enumerate(hits):
        block = [f"[{i+1}] Title: {h['title']}", f"By: {h['by']} · {h['score']} points"]
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

@app.get("/health")
def health():
    return {"ok": True, "mode": RETRIEVAL_MODE, "vectors": state["index"].ntotal}

@app.get("/search")
def search(q: str, k: int = Query(10, ge=1, le=30)):
    return hybrid_search(q, k=k)

@app.get("/ask/stream")
def ask_stream(q: str, k: int = Query(8, ge=1, le=30)):
    hits = hybrid_search(q, k=k)
    prompt = build_prompt(q, hits)

    async def gen():
        yield json.dumps({"type": "sources", "data": hits}) + "\n"
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json={
                    "model": OLLAMA_MODEL, "messages": [{"role": "user", "content": prompt}],
                    "stream": True, "options": {"num_predict": 512, "temperature": 0.3}
                }) as r:
                    async for line in r.aiter_lines():
                        if not line.strip(): continue
                        try: obj = json.loads(line)
                        except: continue
                        chunk = obj.get("message", {}).get("content", "")
                        if chunk: yield json.dumps({"type": "token", "data": chunk}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")