"""SQLite persistence: conversation logs, per-session context and user feedback."""
import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.environ.get("CHATBOT_DB", str(Path(__file__).resolve().parent / "chatbot.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    user_message TEXT NOT NULL,
    bot_reply TEXT NOT NULL,
    intent TEXT NOT NULL,
    faq_id TEXT,
    category TEXT,
    confidence REAL NOT NULL,
    is_fallback INTEGER NOT NULL DEFAULT 0,
    used_context INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_fallback ON messages(is_fallback);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages(id),
    helpful INTEGER NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def log_message(session_id, user_message, result):
    """Store one exchange and return its row id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO messages
               (session_id, created_at, user_message, bot_reply, intent, faq_id,
                category, confidence, is_fallback, used_context)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id,
                _now(),
                user_message,
                result["reply"],
                result["intent"],
                result.get("faq_id"),
                result.get("category"),
                float(result["confidence"]),
                int(result["fallback"]),
                int(result.get("used_context", False)),
            ),
        )
        return cur.lastrowid


def get_last_turn(session_id):
    """Most recent exchange in a session (used for follow-up questions)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def get_history(session_id, limit=50):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, created_at, user_message, bot_reply, intent, confidence
               FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def message_exists(message_id):
    with get_conn() as conn:
        return conn.execute("SELECT 1 FROM messages WHERE id=?", (message_id,)).fetchone() is not None


def add_feedback(message_id, helpful, comment=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (message_id, helpful, comment, created_at) VALUES (?,?,?,?)",
            (message_id, int(helpful), comment, _now()),
        )


def recent_logs(limit=100, only_fallback=False):
    sql = """SELECT m.*, f.helpful FROM messages m
             LEFT JOIN feedback f ON f.message_id = m.id"""
    if only_fallback:
        sql += " WHERE m.is_fallback = 1"
    sql += " ORDER BY m.id DESC LIMIT ?"
    with get_conn() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def stats():
    with get_conn() as conn:
        total, sessions, fallbacks, avg_conf, ctx = conn.execute(
            """SELECT COUNT(*), COUNT(DISTINCT session_id), COALESCE(SUM(is_fallback),0),
                      COALESCE(AVG(confidence),0), COALESCE(SUM(used_context),0)
               FROM messages"""
        ).fetchone()
        top = conn.execute(
            """SELECT faq_id, COUNT(*) AS hits FROM messages
               WHERE faq_id IS NOT NULL GROUP BY faq_id ORDER BY hits DESC LIMIT 5"""
        ).fetchall()
        fb = conn.execute(
            "SELECT COALESCE(SUM(helpful),0), COUNT(*) FROM feedback"
        ).fetchone()
    return {
        "total_messages": total,
        "unique_sessions": sessions,
        "fallback_rate": round(fallbacks / total, 3) if total else 0.0,
        "avg_confidence": round(avg_conf, 3),
        "context_resolved_messages": ctx,
        "top_faqs": [dict(r) for r in top],
        "feedback_helpful": fb[0],
        "feedback_total": fb[1],
    }
