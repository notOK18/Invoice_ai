"""SQLite store for invoices, extractions and review outcomes.

This is the memory the confidence model learns from. Every extracted field is
recorded with the features and confidence it was scored with, and every review
records whether a human accepted that value or corrected it. Those accept/edit
outcomes are the training labels:

    accepted unchanged -> the extraction was right (label 0)
    edited             -> the extraction was wrong (label 1)

Both halves matter. Logging only corrections would teach the model what failure
looks like but never what success looks like.

Nothing here imports the pipeline or sklearn, so the schema stays independent of
how extraction and scoring happen to work.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "invoice_ai.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS invoices (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,              -- file name the invoice came from
    ocr_text    TEXT,                       -- text extraction actually ran on
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extractions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id    INTEGER NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    field         TEXT NOT NULL,            -- invoice_number, total_amount, ...
    value         TEXT,                     -- what the extractor produced
    confidence    REAL,                     -- score at the time it was shown
    model_version INTEGER,                  -- NULL while still on the heuristic
    features      TEXT NOT NULL,            -- JSON: the exact inputs scored
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    extraction_id  INTEGER NOT NULL REFERENCES extractions(id) ON DELETE CASCADE,
    action         TEXT NOT NULL CHECK (action IN ('accepted', 'edited')),
    original_value TEXT,
    final_value    TEXT,
    reviewed_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    version    INTEGER PRIMARY KEY AUTOINCREMENT,
    field      TEXT NOT NULL,
    n_samples  INTEGER NOT NULL,
    n_wrong    INTEGER NOT NULL,
    metrics    TEXT,                        -- JSON
    trained_at TEXT NOT NULL
);

-- Labels inferred from corrections: when a reviewer fixes a field, the word next
-- to the value they typed is a candidate label for that field ("Gesamtbetrag"
-- for a total). A label is only trusted once it has been seen on several
-- different invoices, so a one-off coincidence never becomes a rule.
CREATE TABLE IF NOT EXISTS learned_patterns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    field      TEXT NOT NULL,
    label      TEXT NOT NULL,
    times_seen INTEGER NOT NULL DEFAULT 1,
    active     INTEGER NOT NULL DEFAULT 1,   -- 0 = disabled by a human
    first_seen TEXT NOT NULL,
    last_seen  TEXT NOT NULL,
    UNIQUE (field, label)
);

CREATE INDEX IF NOT EXISTS idx_extractions_invoice ON extractions(invoice_id);
CREATE INDEX IF NOT EXISTS idx_extractions_field   ON extractions(field);
CREATE INDEX IF NOT EXISTS idx_reviews_extraction  ON reviews(extraction_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thin, explicit wrapper over the SQLite file (no ORM, no magic)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def add_invoice(self, source: str, ocr_text: Optional[str] = None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO invoices (source, ocr_text, uploaded_at) VALUES (?, ?, ?)",
                (source, ocr_text, _now()),
            )
            return cur.lastrowid

    def add_extraction(self, invoice_id: int, field: str, value: Optional[str],
                       confidence: Optional[float], features: Dict[str, Any],
                       model_version: Optional[int] = None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """INSERT INTO extractions
                   (invoice_id, field, value, confidence, model_version, features, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (invoice_id, field, value, confidence, model_version,
                 json.dumps(features), _now()),
            )
            return cur.lastrowid

    def record_review(self, extraction_id: int, action: str,
                      original_value: Optional[str], final_value: Optional[str]) -> int:
        """Record that a human accepted or edited one extracted field."""
        if action not in ("accepted", "edited"):
            raise ValueError(f"action must be 'accepted' or 'edited', got {action!r}")
        with self.connect() as conn:
            cur = conn.execute(
                """INSERT INTO reviews
                   (extraction_id, action, original_value, final_value, reviewed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (extraction_id, action, original_value, final_value, _now()),
            )
            return cur.lastrowid

    def record_model(self, field: str, n_samples: int, n_wrong: int,
                     metrics: Optional[Dict[str, Any]] = None) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """INSERT INTO models (field, n_samples, n_wrong, metrics, trained_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (field, n_samples, n_wrong, json.dumps(metrics or {}), _now()),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def training_rows(self, field: str) -> List[Dict[str, Any]]:
        """Reviewed extractions for one field, as {features, label} rows.

        label 1 = the reviewer edited it (the extraction was wrong)
        label 0 = the reviewer accepted it unchanged (it was right)
        """
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT e.features, r.action
                     FROM reviews r
                     JOIN extractions e ON e.id = r.extraction_id
                    WHERE e.field = ?
                 ORDER BY r.id""",
                (field,),
            ).fetchall()
        return [{"features": json.loads(r["features"]),
                 "label": 1 if r["action"] == "edited" else 0} for r in rows]

    def review_counts(self, field: Optional[str] = None) -> Dict[str, int]:
        """How many accepts/edits have been logged (overall or for one field)."""
        query = """SELECT r.action, COUNT(*) AS n
                     FROM reviews r JOIN extractions e ON e.id = r.extraction_id"""
        params: Iterable[Any] = ()
        if field:
            query += " WHERE e.field = ?"
            params = (field,)
        query += " GROUP BY r.action"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        counts = {r["action"]: r["n"] for r in rows}
        counts["total"] = sum(counts.values())
        return counts

    def latest_model_version(self, field: str) -> Optional[int]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MAX(version) AS v FROM models WHERE field = ?", (field,)
            ).fetchone()
        return row["v"] if row and row["v"] is not None else None

    # ------------------------------------------------------------------
    # labels learned from corrections
    # ------------------------------------------------------------------
    def note_label(self, field: str, label: str) -> int:
        """Record that `label` was seen next to a corrected value for `field`.

        Returns how many times it has now been seen.
        """
        label = " ".join(str(label).split())
        if not label:
            return 0
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO learned_patterns (field, label, first_seen, last_seen)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(field, label) DO UPDATE SET
                       times_seen = times_seen + 1,
                       last_seen  = excluded.last_seen""",
                (field, label, _now(), _now()),
            )
            row = conn.execute(
                "SELECT times_seen FROM learned_patterns WHERE field = ? AND label = ?",
                (field, label),
            ).fetchone()
        return row["times_seen"]

    def trusted_labels(self, min_seen: int = 3) -> Dict[str, List[str]]:
        """{field: [label, ...]} for labels confirmed often enough to use."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT field, label FROM learned_patterns
                    WHERE active = 1 AND times_seen >= ?
                 ORDER BY times_seen DESC, label""",
                (min_seen,),
            ).fetchall()
        out: Dict[str, List[str]] = {}
        for row in rows:
            out.setdefault(row["field"], []).append(row["label"])
        return out

    def all_labels(self) -> List[Dict[str, Any]]:
        """Every candidate label with its counts, for showing/managing in the UI."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT field, label, times_seen, active FROM learned_patterns
                 ORDER BY field, times_seen DESC, label"""
            ).fetchall()
        return [dict(r) for r in rows]

    def set_label_active(self, field: str, label: str, active: bool) -> None:
        """Enable or disable a learned label (to remove a bad rule)."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE learned_patterns SET active = ? WHERE field = ? AND label = ?",
                (1 if active else 0, field, label),
            )

    def reviewed_sources(self) -> set:
        """File names that already have at least one recorded review."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT i.source
                     FROM reviews r
                     JOIN extractions e ON e.id = r.extraction_id
                     JOIN invoices    i ON i.id = e.invoice_id"""
            ).fetchall()
        return {row["source"] for row in rows}

    def stats(self) -> Dict[str, Any]:
        with self.connect() as conn:
            def count(table):
                return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            return {"invoices": count("invoices"), "extractions": count("extractions"),
                    "reviews": count("reviews"), "models": count("models")}
