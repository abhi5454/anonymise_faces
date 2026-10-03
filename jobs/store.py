"""SQLite job store with atomic compare-and-set state transitions."""

import json
import sqlite3
import threading
import time

QUEUED, RUNNING = "QUEUED", "RUNNING"
TERMINAL = ("SAFE", "NEEDS_REVIEW", "FAILED")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  upload_id TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  pipeline_version TEXT NOT NULL,
  config_json TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  input_path TEXT,
  output_path TEXT,
  output_sha256 TEXT,
  decision TEXT,
  decision_reasons_json TEXT,
  frame_counts_json TEXT,
  verify_json TEXT,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  old_status TEXT,
  new_status TEXT NOT NULL,
  at REAL NOT NULL,
  note TEXT
);
"""


class JobStore:
    """Thread-safe SQLite store (one connection per call, guarded by a lock)."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def create_or_get(self, job_id, upload_id, content_sha256,
                      pipeline_version, config_json, idempotency_key,
                      input_path):
        """Insert a job, or return the existing row on idempotency clash."""
        now = time.time()
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO jobs (job_id, upload_id, content_sha256,"
                    " pipeline_version, config_json, idempotency_key, status,"
                    " input_path, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (job_id, upload_id, content_sha256, pipeline_version,
                     config_json, idempotency_key, QUEUED, input_path,
                     now, now),
                )
                conn.execute(
                    "INSERT INTO job_events (job_id, old_status, new_status,"
                    " at, note) VALUES (?,?,?,?,?)",
                    (job_id, None, QUEUED, now, "created"),
                )
                duplicate = False
            except sqlite3.IntegrityError:
                duplicate = True
            row = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            return dict(row), duplicate

    def get(self, job_id):
        """Return one job row as a dict, or None."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row) if row else None


    def claim(self, job_id):
        """Atomically QUEUED -> RUNNING. True only if this caller won."""
        return self.transition(job_id, QUEUED, RUNNING, note="claimed")

    def transition(self, job_id, expected, new, note=""):
        """Atomic compare-and-set status transition."""
        now = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ?"
                " WHERE job_id = ? AND status = ?",
                (new, now, job_id, expected),
            )
            if cur.rowcount != 1:
                return False
            conn.execute(
                "INSERT INTO job_events (job_id, old_status, new_status,"
                " at, note) VALUES (?,?,?,?,?)",
                (job_id, expected, new, now, note),
            )
            return True

    def finish(self, job_id, status, **fields):
        """Write result fields then atomically RUNNING -> terminal."""
        allowed = {"output_path", "output_sha256", "decision",
                   "decision_reasons_json", "frame_counts_json",
                   "verify_json", "error", "attempt_count"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        cols = ", ".join(["%s = ?" % k for k in updates] + ["updated_at = ?"])
        now = time.time()
        with self._lock, self._connect() as conn:
            if updates:
                conn.execute("UPDATE jobs SET %s WHERE job_id = ?" % cols,
                             (*updates.values(), now, job_id))
            cur = conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ?"
                " WHERE job_id = ? AND status = ?",
                (status, now, job_id, RUNNING))
            if cur.rowcount != 1:
                return False
            conn.execute(
                "INSERT INTO job_events (job_id, old_status, new_status,"
                " at, note) VALUES (?,?,?,?,?)",
                (job_id, RUNNING, status, now, "finished"))
            return True

    def count_jobs(self):
        """Total job rows (used by the idempotency test)."""
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def public_view(self, row):
        """Shape a DB row into the GET /jobs/{id} response body."""
        for key in ("decision_reasons_json", "frame_counts_json",
                    "verify_json"):
            try:
                row[key] = json.loads(row[key]) if row[key] else None
            except (TypeError, ValueError):
                pass
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "upload_id": row["upload_id"],
            "content_sha256": row["content_sha256"],
            "pipeline_version": row["pipeline_version"],
            "decision": row["decision"],
            "decision_reasons": row["decision_reasons_json"],
            "artifacts": {
                "input_path": row["input_path"],
                "output_path": row["output_path"],
                "output_sha256": row["output_sha256"],
            },
            "frame_counts": row["frame_counts_json"],
            "verify": row["verify_json"],
            "error": row["error"],
        }
