# ingest_core.py
"""
Generic ingest pipeline: RawDoc -> chunks -> embed -> upsert -> prune.
Handles incremental embedding (hash-based) and stale-chunk pruning with FAISS rebuild.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urlparse

import faiss
from sentence_transformers import SentenceTransformer

from db import get_db, migrate

log = logging.getLogger("ingest_core")

INDEX_PATH = os.getenv("HN_INDEX", "hn.faiss")
IDS_PATH   = os.getenv("HN_IDS", "ids.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
PRUNE_REBUILD_THRESHOLD = float(os.getenv("PRUNE_REBUILD_THRESHOLD", "0.05"))


# ---------- HTML helpers ----------
class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)
    def text(self):
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    p = _HTMLStripper()
    try:
        p.feed(html)
    except Exception:
        return ""
    return p.text()


def doc_hash(doc: str) -> str:
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------- Data contract ----------
@dataclass
class RawComment:
    id: int
    parent_id: int | None
    by: str
    text: str
    depth: int = 0


@dataclass
class RawDoc:
    source_id: str
    native_id: int | str
    title: str
    url: str = ""
    author: str = ""
    score: int = 0
    num_comments: int = 0
    published_at: int = 0
    text: str = ""
    kind: str = "story"
    extra: dict = field(default_factory=dict)
    comments: list[RawComment] = field(default_factory=list)


# ---------- Chunking ----------
def chunk_doc(doc: RawDoc) -> list[dict]:
    chunks = []
    body = f"Title: {doc.title}\n\n{doc.text}".strip()
    chunks.append({
        "story_id": doc.native_id,
        "chunk_idx": 0,
        "chunk_kind": "title",
        "parent_id": None,
        "depth": 0,
        "doc": body,
        "hash": doc_hash(body),
        "token_count": approx_tokens(body),
    })
    for j, c in enumerate(doc.comments, 1):
        text = f"Title: {doc.title} (Comment by {c.by}): {c.text}"
        chunks.append({
            "story_id": doc.native_id,
            "chunk_idx": j,
            "chunk_kind": "comment",
            "parent_id": c.parent_id,
            "depth": c.depth,
            "doc": text,
            "hash": doc_hash(text),
            "token_count": approx_tokens(text),
        })
    return chunks


# ---------- FAISS helpers ----------
def _load_index(dim: int):
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    log.info("Creating new FAISS index (dim=%d)", dim)
    return faiss.IndexFlatIP(dim)


def _atomic_write_index(index):
    tmp = INDEX_PATH + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, INDEX_PATH)


def _atomic_write_ids(ids):
    tmp = IDS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ids, f)
    os.replace(tmp, IDS_PATH)


def _domain_of(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
        return host.removeprefix("www.")
    except Exception:
        return ""


# ---------- Upsert ----------
def _upsert_stories(conn, docs: Iterable[RawDoc]):
    now = int(time.time())
    rows = []
    for d in docs:
        rows.append((
            d.native_id, d.title, d.url, d.author, d.score, d.num_comments,
            d.published_at, d.text, json.dumps([c.__dict__ for c in d.comments]),
            "",
            d.kind,
            _domain_of(d.url),
            doc_hash(f"{d.title}|{d.text}|{d.published_at}"),
            now, now,
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO stories
           (id, title, url, by, score, descendants, time, text, comments_json, doc_hash,
            type, domain, story_hash, fetched_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


# ---------- Prune + rebuild ----------
def _prune_stale_chunks(conn, current_keys: set[tuple[int, int]]) -> int:
    existing = {tuple(r) for r in conn.execute(
        "SELECT story_id, chunk_idx FROM chunks")}
    stale = existing - current_keys
    if not stale:
        return 0
    log.info("Pruning %d stale chunks", len(stale))
    conn.executemany(
        "DELETE FROM chunks WHERE story_id=? AND chunk_idx=?", list(stale))
    return len(stale)


def _rebuild_faiss(conn, model: SentenceTransformer):
    survivors = [tuple(r) for r in conn.execute(
        "SELECT story_id, chunk_idx FROM chunks ORDER BY story_id, chunk_idx")]
    if not survivors:
        log.warning("No chunks remain; writing empty index")
        dim = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)
        _atomic_write_index(index)
        conn.execute("DELETE FROM chunk_vectors")
        conn.commit()
        _atomic_write_ids([])
        return

    docs = []
    for sid, cidx in survivors:
        row = conn.execute(
            "SELECT doc FROM chunks WHERE story_id=? AND chunk_idx=?",
            (sid, cidx)).fetchone()
        docs.append(row["doc"])
    vecs = model.encode(docs, normalize_embeddings=True,
                        batch_size=32, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    _atomic_write_index(index)

    conn.execute("DELETE FROM chunk_vectors")
    conn.executemany(
        "INSERT INTO chunk_vectors (faiss_row, story_id, chunk_idx) VALUES (?,?,?)",
        [(i, sid, cidx) for i, (sid, cidx) in enumerate(survivors)])
    conn.commit()
    _atomic_write_ids([[sid, cidx] for sid, cidx in survivors])
    log.info("FAISS rebuilt with %d vectors", index.ntotal)


# ---------- Main entry ----------
def ingest_docs(docs: list[RawDoc], model: SentenceTransformer | None = None):
    migrate()

    if not docs:
        log.info("No docs to ingest")
        return

    if model is None:
        model = SentenceTransformer(EMBED_MODEL)

    with get_db() as conn:
        _upsert_stories(conn, docs)

        all_chunks = []
        for d in docs:
            all_chunks.extend(chunk_doc(d))
        current_keys = {(c["story_id"], c["chunk_idx"]) for c in all_chunks}

        n_pruned = _prune_stale_chunks(conn, current_keys)
        conn.commit()

        existing = {(r["story_id"], r["chunk_idx"]): r["doc_hash"]
                    for r in conn.execute("SELECT story_id, chunk_idx, doc_hash FROM chunks")}
        to_embed = [c for c in all_chunks
                    if existing.get((c["story_id"], c["chunk_idx"])) != c["hash"]]

        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        rebuild_needed = (
            n_pruned > 0 and
            (total_chunks == 0 or n_pruned / max(total_chunks, 1) > PRUNE_REBUILD_THRESHOLD)
        )

        if to_embed:
            conn.executemany(
                """INSERT OR REPLACE INTO chunks
                   (story_id, chunk_idx, chunk_kind, parent_id, depth, doc, doc_hash, token_count)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [(c["story_id"], c["chunk_idx"], c["chunk_kind"], c["parent_id"],
                  c["depth"], c["doc"], c["hash"], c["token_count"]) for c in to_embed])
            conn.commit()

        if rebuild_needed:
            log.info("Rebuilding FAISS (pruned %d / %d)", n_pruned, total_chunks)
            _rebuild_faiss(conn, model)
            return

        if not to_embed and n_pruned == 0:
            log.info("No changes; nothing to embed")
            return

        if to_embed:
            log.info("Embedding %d new chunks", len(to_embed))
            vecs = model.encode(
                [c["doc"] for c in to_embed],
                normalize_embeddings=True, batch_size=32,
                show_progress_bar=False).astype("float32")

            index = _load_index(vecs.shape[1])
            start_row = index.ntotal
            index.add(vecs)
            _atomic_write_index(index)

            conn.executemany(
                "INSERT INTO chunk_vectors (faiss_row, story_id, chunk_idx) VALUES (?,?,?)",
                [(start_row + i, c["story_id"], c["chunk_idx"]) for i, c in enumerate(to_embed)])
            conn.commit()

            legacy = [[r["story_id"], r["chunk_idx"]] for r in conn.execute(
                "SELECT story_id, chunk_idx FROM chunk_vectors ORDER BY faiss_row")]
            _atomic_write_ids(legacy)

            log.info("Index now has %d vectors", index.ntotal)