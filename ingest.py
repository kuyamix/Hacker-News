# ingest.py
"""HN adapter. Fetches stories + top-N comments, hands off to ingest_core."""
import os
import time
import logging
import requests

from db import migrate
from ingest_core import RawDoc, RawComment, html_to_text, ingest_docs

BASE = "https://hacker-news.firebaseio.com/v0"
LIMIT = int(os.getenv("INGEST_LIMIT", "100"))
COMMENTS_PER_STORY = int(os.getenv("COMMENTS_PER_STORY", "5"))
FEEDS = os.getenv("INGEST_FEEDS", "topstories").split(",")
HN_DEPTH = int(os.getenv("HN_COMMENT_DEPTH", "1"))

log = logging.getLogger("ingest_hn")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def fetch_json(url: str):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def fetch_comments(story: dict, n: int = COMMENTS_PER_STORY,
                   depth: int = HN_DEPTH) -> list[RawComment]:
    out: list[RawComment] = []
    queue = [(k, None, 0) for k in (story.get("kids") or [])[:n]]
    while queue and len(out) < n:
        kid, parent, d = queue.pop(0)
        try:
            c = fetch_json(f"{BASE}/item/{kid}.json")
        except Exception:
            continue
        if not c or c.get("deleted") or c.get("dead"):
            continue
        text = html_to_text(c.get("text", ""))
        if not text:
            continue
        out.append(RawComment(
            id=c["id"], parent_id=parent, by=c.get("by", ""),
            text=text, depth=d,
        ))
        if d + 1 < depth:
            for sub in (c.get("kids") or [])[:2]:
                queue.append((sub, c["id"], d + 1))
    return out


def fetch_hn_docs() -> list[RawDoc]:
    ids = []
    for feed in FEEDS:
        feed = feed.strip()
        log.info("Fetching %s ids…", feed)
        ids.extend(fetch_json(f"{BASE}/{feed}.json")[:LIMIT])
    ids = list(dict.fromkeys(ids))

    docs: list[RawDoc] = []
    for item_id in ids:
        try:
            item = fetch_json(f"{BASE}/item/{item_id}.json")
        except Exception:
            continue
        if not item or not item.get("title"):
            continue
        docs.append(RawDoc(
            source_id="hn",
            native_id=item["id"],
            title=item["title"],
            url=item.get("url", ""),
            author=item.get("by", ""),
            score=item.get("score", 0),
            num_comments=item.get("descendants", 0),
            published_at=item.get("time", 0),
            text=html_to_text(item.get("text", "")),
            kind="story",
            extra={"feed": item.get("type", "story")},
            comments=fetch_comments(item),
        ))
        time.sleep(0.03)
    return docs


def main():
    migrate()
    docs = fetch_hn_docs()
    log.info("Fetched %d docs; handing to ingest_core", len(docs))
    ingest_docs(docs)


if __name__ == "__main__":
    main()
