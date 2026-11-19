# migrate_v1_to_v2.py
"""
One-off: upgrade an existing v1 hn.db to v2.

Steps:
  1. Apply migrations/001_v2_schema.sql (via db.migrate).
  2. Populate chunk_vectors from ids.json.
  3. Verify consistency: FAISS ntotal == chunk_vectors rows.
  4. Backfill stories.domain from stories.url.
"""
import json
import logging
import os
from urllib.parse import urlparse

import faiss

from db import get_db, migrate

log = logging.getLogger("migrate_v1_to_v2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Paths live here (not in db.py) — matches ingest_core.py / server.py.
INDEX_PATH = os.getenv("HN_INDEX", "hn.faiss")
IDS_PATH   = os.getenv("HN_IDS", "ids.json")


def main():
    migrate()   # applies migrations/001_v2_schema.sql if user_version < 1

    if not os.path.exists(IDS_PATH):
        log.warning("%s not found; skipping chunk_vectors population", IDS_PATH)
    else:
        with open(IDS_PATH) as f:
            ids = json.load(f)

        with get_db() as conn:
            n = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
            if n > 0:
                log.info("chunk_vectors already has %d rows; skipping", n)
            else:
                log.info("Populating chunk_vectors from %s (%d rows)", IDS_PATH, len(ids))
                conn.executemany(
                    "INSERT INTO chunk_vectors (faiss_row, story_id, chunk_idx) VALUES (?,?,?)",
                    [(i, int(pair[0]), int(pair[1])) for i, pair in enumerate(ids)],
                )
                conn.commit()

            # Backfill domain
            rows = conn.execute(
                "SELECT id, url FROM stories WHERE domain IS NULL OR domain = ''"
            ).fetchall()
            updates = []
            for r in rows:
                try:
                    host = (urlparse(r["url"]).hostname or "").removeprefix("www.")
                except Exception:
                    host = ""
                updates.append((host, r["id"]))
            if updates:
                conn.executemany("UPDATE stories SET domain=? WHERE id=?", updates)
                conn.commit()
                log.info("Backfilled domain for %d stories", len(updates))

    # Sanity check
    if not os.path.exists(INDEX_PATH):
        log.error("FAISS index %s not found — cannot verify consistency.", INDEX_PATH)
        return

    index = faiss.read_index(INDEX_PATH)
    with get_db() as conn:
        n_cv     = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_fts    = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    log.info(
        "FAISS ntotal=%d, chunk_vectors=%d, chunks=%d, chunks_fts=%d",
        index.ntotal, n_cv, n_chunks, n_fts,
    )
    if index.ntotal != n_cv:
        log.error("MISMATCH: FAISS ntotal != chunk_vectors. Do NOT start the server yet.")
    else:
        log.info("Consistent. Safe to start server.")


if __name__ == "__main__":
    main()