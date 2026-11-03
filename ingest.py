# ingest.py
"""
Fetches HN stories + top comments, embeds incremental docs, and appends
to FAISS. Idempotent: re-running only embeds new or changed stories.
"""
import json, time, sqlite3, requests, re, os
from html.parser import HTMLParser
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = "https://hacker-news.firebaseio.com/v0"
LIMIT = int(os.getenv("INGEST_LIMIT", "100"))
COMMENTS_PER_STORY = 5
FEEDS = ["topstories"]  # add "beststories", "newstories" for broader corpus


def fetch_json(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)
    def text(self):
        return re.sub(r"\s+", " ", "".join(self.parts)).strip()


def html_to_text(html):
    if not html:
        return ""
    p = _HTMLStripper()
    try:
        p.feed(html)
    except Exception:
        return ""
    return p.text()


def fetch_comments(story, n=COMMENTS_PER_STORY):
    """Fetch top n non-deleted comment texts (depth-1) for a story."""
    out = []
    for kid in (story.get("kids") or [])[:n]:
        try:
            c = fetch_json(f"{BASE}/item/{kid}.json")
        except Exception:
            continue
        if not c or c.get("deleted") or c.get("dead"):
            continue
        t = html_to_text(c.get("text", ""))
        if t:
            out.append({"by": c.get("by", ""), "text": t})
    return out


def build_doc(title, self_text, comments):
    """The document we embed. Title is always first so it's weighted."""
    parts = [title]
    if self_text:
        parts.append(self_text)
    for c in comments:
        parts.append(f"(comment by {c['by']}): {c['text']}")
    return "\n\n".join(parts)


def get_db():
    conn = sqlite3.connect("hn.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS stories (
        id INTEGER PRIMARY KEY,
        title TEXT,
        url TEXT,
        by TEXT,
        score INTEGER,
        descendants INTEGER,
        time INTEGER,
        text TEXT,
        comments_json TEXT,
        doc_hash TEXT
    )""")
    # Migration for older DBs created before the column additions.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(stories)")}
    for col, decl in [
        ("descendants", "INTEGER"), ("time", "INTEGER"),
        ("comments_json", "TEXT"), ("doc_hash", "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE stories ADD COLUMN {col} {decl}")
    return conn


def main():
    conn = get_db()
    ids = []
    for feed in FEEDS:
        print(f"Fetching {feed} ids…")
        ids.extend(fetch_json(f"{BASE}/{feed}.json")[:LIMIT])
    ids = list(dict.fromkeys(ids))  # dedupe preserving order

    print(f"Have {len(ids)} candidate ids. Fetching story details…")
    stories = []
    for i, item_id in enumerate(ids, 1):
        try:
            item = fetch_json(f"{BASE}/item/{item_id}.json")
        except Exception as e:
            print(f"  skip {item_id}: {e}")
            continue
        if not item or not item.get("title"):
            continue
        comments = fetch_comments(item)
        doc = build_doc(item["title"], html_to_text(item.get("text", "")), comments)
        doc_hash = str(hash(doc))
        stories.append({
            "id": item["id"],
            "title": item["title"],
            "url": item.get("url", ""),
            "by": item.get("by", ""),
            "score": item.get("score", 0),
            "descendants": item.get("descendants", 0),
            "time": item.get("time", 0),
            "text": html_to_text(item.get("text", "")),
            "comments": comments,
            "doc": doc,
            "doc_hash": doc_hash,
        })
        print(f"  [{i}/{len(ids)}] {item['title'][:70]}")
        time.sleep(0.03)

    # Which stories actually need embedding?
    existing_hashes = {
        row[0]: row[1]
        for row in conn.execute("SELECT id, doc_hash FROM stories WHERE doc_hash IS NOT NULL")
    }
    to_embed = [s for s in stories if existing_hashes.get(s["id"]) != s["doc_hash"]]
    print(f"{len(stories)} fetched, {len(to_embed)} need embedding.")

    # Persist metadata for every story (cheap).
    conn.executemany(
        """INSERT OR REPLACE INTO stories
           (id, title, url, by, score, descendants, time, text, comments_json, doc_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [(s["id"], s["title"], s["url"], s["by"], s["score"], s["descendants"],
          s["time"], s["text"], json.dumps(s["comments"]), s["doc_hash"])
         for s in stories],
    )
    conn.commit()

    if not to_embed:
        print("Nothing new to embed. Metadata updated.")
        conn.close()
        return

    print("Loading model…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(
        [s["doc"] for s in to_embed],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    ).astype("float32")

    # Load existing index, or create. Index is flat (exact search); appending is O(1).
    if os.path.exists("hn.faiss") and os.path.exists("ids.json"):
        index = faiss.read_index("hn.faiss")
        with open("ids.json") as f:
            existing_ids = json.load(f)
        # Drop stale entries whose ids we no longer know about (none normally).
        assert index.ntotal == len(existing_ids), "index/ids.json desync"
    else:
        index = faiss.IndexFlatIP(vecs.shape[1])
        existing_ids = []

    # Only append ids for newly embedded stories. Re-embedded stories
    # (doc_hash changed) are appended too — we accept a small duplicate
    # in the index and dedupe at query time by keeping the best score per id.
    existing_ids.extend(s["id"] for s in to_embed)
    index.add(vecs)

    # Atomic-ish writes: write temp then rename.
    faiss.write_index(index, "hn.faiss.tmp")
    os.replace("hn.faiss.tmp", "hn.faiss")
    with open("ids.json.tmp", "w") as f:
        json.dump(existing_ids, f)
    os.replace("ids.json.tmp", "ids.json")

    # Guard against runaway duplicates: if >30% of the index is stale, rebuild.
    uniq = len(set(existing_ids))
    if len(existing_ids) > uniq * 1.3:
        print(f"Compacting index ({len(existing_ids)} -> {uniq})…")
        _compact(model, conn)

    conn.close()
    print(f"Wrote hn.faiss ({index.ntotal} vectors), ids.json, hn.db")


def _compact(model, conn):
    """Rebuild FAISS from current stories table, deduped by id."""
    rows = list(conn.execute(
        "SELECT id, title, text, comments_json FROM stories ORDER BY id"))
    docs, ids = [], []
    for sid, title, text, comments_json in rows:
        comments = json.loads(comments_json or "[]")
        docs.append(build_doc(title, text, comments))
        ids.append(sid)
    vecs = model.encode(docs, normalize_embeddings=True).astype("float32")
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    faiss.write_index(idx, "hn.faiss.tmp")
    os.replace("hn.faiss.tmp", "hn.faiss")
    with open("ids.json.tmp", "w") as f:
        json.dump(ids, f)
    os.replace("ids.json.tmp", "ids.json")


if __name__ == "__main__":
    main()