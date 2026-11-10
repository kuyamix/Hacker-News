# ingest.py
import json, time, sqlite3, requests, re, os, hashlib
from html.parser import HTMLParser
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = "https://hacker-news.firebaseio.com/v0"
LIMIT = int(os.getenv("INGEST_LIMIT", "100"))
COMMENTS_PER_STORY = 5
FEEDS = ["topstories"] 

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
    if not html: return ""
    p = _HTMLStripper()
    try: p.feed(html)
    except Exception: return ""
    return p.text()

def fetch_comments(story, n=COMMENTS_PER_STORY):
    out = []
    for kid in (story.get("kids") or [])[:n]:
        try:
            c = fetch_json(f"{BASE}/item/{kid}.json")
        except Exception:
            continue
        if not c or c.get("deleted") or c.get("dead"): continue
        t = html_to_text(c.get("text", ""))
        if t: out.append({"by": c.get("by", ""), "text": t})
    return out

def doc_hash(doc: str) -> str:
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()

def get_db():
    conn = sqlite3.connect("hn.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS stories (
        id INTEGER PRIMARY KEY, title TEXT, url TEXT, by TEXT, score INTEGER,
        descendants INTEGER, time INTEGER, text TEXT, comments_json TEXT, doc_hash TEXT
    )""")
    # NEW: Chunks table for passage-level retrieval
    conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
        story_id INTEGER, chunk_idx INTEGER, doc TEXT, doc_hash TEXT,
        PRIMARY KEY (story_id, chunk_idx)
    )""")
    return conn

def main():
    conn = get_db()
    ids = []
    for feed in FEEDS:
        print(f"Fetching {feed} ids…")
        ids.extend(fetch_json(f"{BASE}/{feed}.json")[:LIMIT])
    ids = list(dict.fromkeys(ids))

    print(f"Have {len(ids)} candidate ids. Fetching story details…")
    stories = []
    for i, item_id in enumerate(ids, 1):
        try: item = fetch_json(f"{BASE}/item/{item_id}.json")
        except Exception: continue
        if not item or not item.get("title"): continue
        
        comments = fetch_comments(item)
        self_text = html_to_text(item.get("text", ""))
        
        stories.append({
            "id": item["id"], "title": item["title"], "url": item.get("url", ""),
            "by": item.get("by", ""), "score": item.get("score", 0),
            "descendants": item.get("descendants", 0), "time": item.get("time", 0),
            "text": self_text, "comments": comments,
        })
        time.sleep(0.03)

    # Prepare chunks for embedding
    all_chunks = []
    for s in stories:
        # Chunk 0: Title + Self Text
        c0 = f"Title: {s['title']}\n\n{s['text']}"
        all_chunks.append({"story_id": s["id"], "chunk_idx": 0, "doc": c0, "hash": doc_hash(c0)})
        
        # Chunk 1..N: Comments
        for j, c in enumerate(s["comments"], 1):
            c_doc = f"Title: {s['title']} (Comment by {c['by']}): {c['text']}"
            all_chunks.append({"story_id": s["id"], "chunk_idx": j, "doc": c_doc, "hash": doc_hash(c_doc)})

    # Find which chunks need embedding
    existing_chunks = { (r[0], r[1]): r[2] for r in conn.execute("SELECT story_id, chunk_idx, doc_hash FROM chunks") }
    to_embed = [c for c in all_chunks if existing_chunks.get((c["story_id"], c["chunk_idx"])) != c["hash"]]

    # Save metadata
    conn.executemany(
        """INSERT OR REPLACE INTO stories (id, title, url, by, score, descendants, time, text, comments_json, doc_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [(s["id"], s["title"], s["url"], s["by"], s["score"], s["descendants"], s["time"], s["text"], json.dumps(s["comments"]), "") for s in stories]
    )
    conn.commit()

    if not to_embed:
        print("No new chunks to embed.")
        return

    print(f"Embedding {len(to_embed)} chunks…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode([c["doc"] for c in to_embed], normalize_embeddings=True, batch_size=32).astype("float32")

    # Build or load index
    if os.path.exists("hn.faiss") and os.path.exists("ids.json"):
        index = faiss.read_index("hn.faiss")
        with open("ids.json") as f: existing_ids = json.load(f)
    else:
        index = faiss.IndexFlatIP(vecs.shape[1])
        existing_ids = []

    # Save chunk records
    conn.executemany(
        "INSERT OR REPLACE INTO chunks (story_id, chunk_idx, doc, doc_hash) VALUES (?,?,?,?)",
        [(c["story_id"], c["chunk_idx"], c["doc"], c["hash"]) for c in to_embed]
    )
    conn.commit()

    # Append to FAISS (ids are now [[story_id, chunk_idx], ...])
    existing_ids.extend([[c["story_id"], c["chunk_idx"]] for c in to_embed])
    index.add(vecs)
    
    faiss.write_index(index, "hn.faiss.tmp"); os.replace("hn.faiss.tmp", "hn.faiss")
    with open("ids.json.tmp", "w") as f: json.dump(existing_ids, f)
    os.replace("ids.json.tmp", "ids.json")
    
    print(f"Wrote {len(to_embed)} new vectors to index. Total: {index.ntotal}")
    conn.close()

if __name__ == "__main__":
    main()