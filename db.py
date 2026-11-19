# db.py
"""Shared DB connection + migration runner. Used by ingest.py, server.py, migrate_v1_to_v2.py."""
import os
import sqlite3
import logging
from pathlib import Path

DB_PATH = os.getenv("HN_DB", "hn.db")
MIGRATIONS_DIR = Path(__file__).parent / "migrations"

log = logging.getLogger("hn_db")


def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def get_db():
    return _connect()


def current_version(conn) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _set_version(conn, v: int):
    conn.execute(f"PRAGMA user_version = {int(v)}")


def migrate(target: int | None = None):
    """Apply pending migrations in order. Idempotent (tracked by user_version)."""
    conn = _connect()
    try:
        cur = current_version(conn)
        if target is None:
            target = _discover_latest()
        if cur >= target:
            log.info("DB already at version %d", cur)
            return cur

        for v in range(cur + 1, target + 1):
            f = _find_migration(v)
            if f is None:
                raise RuntimeError(f"No migration file for version {v}")
            log.info("Applying migration %s", f.name)
            conn.executescript(f.read_text())
            _set_version(conn, v)
            conn.commit()
        log.info("DB migrated to version %d", target)
        return target
    finally:
        conn.close()


def _discover_latest() -> int:
    if not MIGRATIONS_DIR.exists():
        return 0
    vs = []
    for f in MIGRATIONS_DIR.glob("*.sql"):
        try:
            vs.append(int(f.name.split("_", 1)[0]))
        except ValueError:
            continue
    return max(vs) if vs else 0


def _find_migration(v: int):
    for f in MIGRATIONS_DIR.glob(f"{v:03d}_*.sql"):
        return f
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    migrate()