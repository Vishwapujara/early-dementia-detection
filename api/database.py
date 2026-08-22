"""SQLite-backed prediction logging.

Every inference the API serves is logged here (model, predicted class,
confidence, timestamp), which is what `api/drift.py` reads to compute each
model's recent output distribution. A fresh connection is opened per call
rather than held open — this is a low-traffic demo service, not a
high-throughput one, so the simplicity of not managing a shared connection
pool/thread-safety story outweighs the per-call connection overhead.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "predictions.db"


def get_connection(db_path=None):
    # Resolved inside the function body (not as a `db_path=DEFAULT_DB_PATH`
    # default) so tests can do `database.DEFAULT_DB_PATH = tmp_path` and
    # have every call pick it up -- a plain default argument would freeze
    # in the original value at import time instead.
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )


def log_prediction(model, predicted_class, confidence, db_path=None, timestamp=None):
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, model, predicted_class, confidence) VALUES (?, ?, ?, ?)",
            (timestamp, model, predicted_class, confidence),
        )


def get_recent_predictions(limit=50, model=None, db_path=None):
    query = "SELECT id, timestamp, model, predicted_class, confidence FROM predictions"
    params = []
    if model is not None:
        query += " WHERE model = ?"
        params.append(model)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_class_counts(model, limit=None, db_path=None):
    """Class distribution over the most recent `limit` predictions for `model`
    (or all logged predictions for that model if `limit` is None).
    """
    if limit is None:
        query = "SELECT predicted_class, COUNT(*) as n FROM predictions WHERE model = ? GROUP BY predicted_class"
        params = [model]
    else:
        query = """
            SELECT predicted_class, COUNT(*) as n FROM (
                SELECT predicted_class FROM predictions WHERE model = ? ORDER BY id DESC LIMIT ?
            ) GROUP BY predicted_class
        """
        params = [model, limit]

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return {row["predicted_class"]: row["n"] for row in rows}
