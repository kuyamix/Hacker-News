-- migrations/001_v2_schema.sql
-- Phase 1 schema upgrade. Idempotent-ish: run once via db.py migrate().
-- Assumes the v1 tables (stories, chunks) exist.

PRAGMA foreign_keys = ON;

-- ---- documents (renamed conceptually; keep 'stories' table for compat) ----
-- We keep the name `stories` to avoid breaking existing code mid-phase,
-- but add the new columns. The rename to `documents` can happen in phase 4.

ALTER TABLE stories ADD COLUMN type      TEXT DEFAULT 'story';
ALTER TABLE stories ADD COLUMN domain    TEXT;
ALTER TABLE stories ADD COLUMN story_hash TEXT;
ALTER TABLE stories ADD COLUMN fetched_at INTEGER;
ALTER TABLE stories ADD COLUMN updated_at INTEGER;

CREATE INDEX IF NOT EXISTS idx_stories_time  ON stories(time DESC);
CREATE INDEX IF NOT EXISTS idx_stories_score ON stories(score DESC);
CREATE INDEX IF NOT EXISTS idx_stories_domain ON stories(domain);

-- ---- chunks: add structural columns ----
ALTER TABLE chunks ADD COLUMN chunk_kind  TEXT;      -- 'title' | 'selftext' | 'comment'
ALTER TABLE chunks ADD COLUMN parent_id   INTEGER;   -- comment parent (HN item id)
ALTER TABLE chunks ADD COLUMN depth       INTEGER;   -- comment nesting depth
ALTER TABLE chunks ADD COLUMN token_count INTEGER;

-- Backfill: any existing chunk with chunk_idx=0 is title+selftext; others are comments
UPDATE chunks SET chunk_kind = CASE WHEN chunk_idx = 0 THEN 'title' ELSE 'comment' END
WHERE chunk_kind IS NULL;

-- ---- chunk_vectors: replaces ids.json ----
-- faiss_row is the row index in hn.faiss. doc_id/chunk_idx reconstruct the mapping.
CREATE TABLE IF NOT EXISTS chunk_vectors (
  faiss_row INTEGER PRIMARY KEY,
  story_id  INTEGER NOT NULL,
  chunk_idx INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cv_story ON chunk_vectors(story_id);

-- Populate from existing ids.json is done in Python (migrate_v1_to_v2.py),
-- because SQL can't read JSON files.

-- ---- FTS5 over chunks.doc ----
-- content='chunks' with contentless-delete off; we maintain via triggers.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  doc,
  content='chunks',
  content_rowid='rowid',
  tokenize='porter unicode61'
);

-- Rebuild FTS from existing chunks (safe to run once)
INSERT INTO chunks_fts(rowid, doc) SELECT rowid, doc FROM chunks;

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, doc) VALUES (new.rowid, new.doc);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, doc) VALUES ('delete', old.rowid, old.doc);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, doc) VALUES ('delete', old.rowid, old.doc);
  INSERT INTO chunks_fts(rowid, doc) VALUES (new.rowid, new.doc);
END;