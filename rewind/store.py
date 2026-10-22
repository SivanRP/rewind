"""Durable job store backed by SQLite.

Every state change is a single transaction, so a crash at any point leaves
the queue in a consistent state. Workers claim jobs with a time-limited
lease; if a worker dies, its lease expires and another worker picks the job
up. Writes from a worker whose lease was taken over are rejected (fencing).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    idempotency_key   TEXT UNIQUE,
    model             TEXT NOT NULL,
    messages          TEXT NOT NULL,
    params            TEXT NOT NULL DEFAULT '{}',
    status            TEXT NOT NULL,      -- queued | running | retrying | succeeded | dead
    attempts          INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL,
    next_attempt_at   REAL NOT NULL,
    lease_owner       TEXT,
    lease_expires_at  REAL,
    result            TEXT,
    last_error        TEXT,
    latency_ms        REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          REAL,
    replay_count      INTEGER NOT NULL DEFAULT 0,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    finished_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_ready ON jobs (status, next_attempt_at);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT NOT NULL,
    ts      REAL NOT NULL,
    kind    TEXT NOT NULL,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events (job_id);
"""

TERMINAL = ("succeeded", "dead")


@dataclass
class Job:
    id: str
    idempotency_key: Optional[str]
    model: str
    messages: list
    params: dict
    status: str
    attempts: int
    max_attempts: int
    next_attempt_at: float
    lease_owner: Optional[str]
    lease_expires_at: Optional[float]
    result: Optional[str]
    last_error: Optional[str]
    latency_ms: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cost_usd: Optional[float]
    replay_count: int
    created_at: float
    updated_at: float
    finished_at: Optional[float]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        d = dict(row)
        d["messages"] = json.loads(d["messages"])
        d["params"] = json.loads(d["params"])
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Store:
    def __init__(self, path: str = "rewind.db"):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    # ---- helpers -------------------------------------------------------
    def _event(self, job_id: str, kind: str, detail: str = "") -> None:
        self._conn.execute(
            "INSERT INTO events (job_id, ts, kind, detail) VALUES (?, ?, ?, ?)",
            (job_id, time.time(), kind, detail),
        )

    def _tx(self):
        """BEGIN IMMEDIATE grabs the write lock up front, so two processes
        sharing the same database can never claim the same job."""
        store = self

        class _Tx:
            def __enter__(self_inner):
                store._lock.acquire()
                store._conn.execute("BEGIN IMMEDIATE")
                return store._conn

            def __exit__(self_inner, exc_type, *_):
                try:
                    store._conn.execute("ROLLBACK" if exc_type else "COMMIT")
                finally:
                    store._lock.release()

        return _Tx()

    # ---- producer side -------------------------------------------------
    def enqueue(
        self,
        model: str,
        messages: list,
        params: Optional[dict] = None,
        idempotency_key: Optional[str] = None,
        max_attempts: int = 5,
    ) -> tuple[Job, bool]:
        """Add a job. Returns (job, created). Submitting the same
        idempotency_key twice returns the original job instead of a copy."""
        now = time.time()
        job_id = uuid.uuid4().hex
        with self._tx() as c:
            cur = c.execute(
                """INSERT INTO jobs (id, idempotency_key, model, messages, params, status,
                       max_attempts, next_attempt_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                   ON CONFLICT (idempotency_key) DO NOTHING""",
                (job_id, idempotency_key, model, json.dumps(messages),
                 json.dumps(params or {}), max_attempts, now, now, now),
            )
            created = cur.rowcount == 1
            if created:
                self._event(job_id, "enqueued")
                row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            else:
                row = c.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                self._event(row["id"], "duplicate_ignored", idempotency_key or "")
        return Job.from_row(row), created

    # ---- worker side ---------------------------------------------------
    def claim(self, worker_id: str, lease_seconds: float) -> Optional[Job]:
        """Atomically take the next ready job, or reclaim one whose lease expired."""
        now = time.time()
        with self._tx() as c:
            row = c.execute(
                """SELECT id, status FROM jobs
                   WHERE (status IN ('queued', 'retrying') AND next_attempt_at <= ?)
                      OR (status = 'running' AND lease_expires_at < ?)
                   ORDER BY next_attempt_at
                   LIMIT 1""",
                (now, now),
            ).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                self._event(row["id"], "lease_expired", "previous worker presumed dead")
            c.execute(
                """UPDATE jobs SET status = 'running', lease_owner = ?, lease_expires_at = ?,
                       attempts = attempts + 1, updated_at = ?
                   WHERE id = ?""",
                (worker_id, now + lease_seconds, now, row["id"]),
            )
            self._event(row["id"], "claimed", worker_id)
            claimed = c.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return Job.from_row(claimed)

    def complete(
        self,
        job_id: str,
        worker_id: str,
        result: str,
        latency_ms: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> bool:
        """Mark success. Returns False if this worker no longer holds the lease."""
        now = time.time()
        with self._tx() as c:
            cur = c.execute(
                """UPDATE jobs SET status = 'succeeded', result = ?, latency_ms = ?,
                       prompt_tokens = ?, completion_tokens = ?, cost_usd = ?,
                       lease_owner = NULL, lease_expires_at = NULL, last_error = NULL,
                       updated_at = ?, finished_at = ?
                   WHERE id = ? AND status = 'running' AND lease_owner = ?""",
                (result, latency_ms, prompt_tokens, completion_tokens, cost_usd,
                 now, now, job_id, worker_id),
            )
            ok = cur.rowcount == 1
            self._event(job_id, "succeeded" if ok else "stale_write_rejected", worker_id)
        return ok

    def fail(
        self, job_id: str, worker_id: str, error: str, retryable: bool, retry_in: float
    ) -> Optional[str]:
        """Record a failure. Schedules a retry or moves the job to the dead-letter
        queue. Returns the new status, or None if the lease was lost."""
        now = time.time()
        with self._tx() as c:
            row = c.execute(
                "SELECT attempts, max_attempts FROM jobs WHERE id = ? AND status = 'running' AND lease_owner = ?",
                (job_id, worker_id),
            ).fetchone()
            if row is None:
                self._event(job_id, "stale_write_rejected", worker_id)
                return None
            if retryable and row["attempts"] < row["max_attempts"]:
                status, next_at, finished = "retrying", now + retry_in, None
            else:
                status, next_at, finished = "dead", now, now
            c.execute(
                """UPDATE jobs SET status = ?, last_error = ?, next_attempt_at = ?,
                       lease_owner = NULL, lease_expires_at = NULL,
                       updated_at = ?, finished_at = ?
                   WHERE id = ?""",
                (status, error[:500], next_at, now, finished, job_id),
            )
            detail = f"{error[:200]} (retry in {retry_in:.2f}s)" if status == "retrying" else error[:200]
            self._event(job_id, "retry_scheduled" if status == "retrying" else "dead_lettered", detail)
        return status

    # ---- operator side -------------------------------------------------
    def replay(self, job_ids: Optional[list[str]] = None) -> int:
        """Send dead-lettered jobs back to the queue with a fresh attempt budget."""
        now = time.time()
        with self._tx() as c:
            if job_ids:
                marks = ",".join("?" * len(job_ids))
                ids = [r["id"] for r in c.execute(
                    f"SELECT id FROM jobs WHERE status = 'dead' AND id IN ({marks})", job_ids)]
            else:
                ids = [r["id"] for r in c.execute("SELECT id FROM jobs WHERE status = 'dead'")]
            for job_id in ids:
                c.execute(
                    """UPDATE jobs SET status = 'queued', attempts = 0, next_attempt_at = ?,
                           replay_count = replay_count + 1, finished_at = NULL, updated_at = ?
                       WHERE id = ?""",
                    (now, now, job_id),
                )
                self._event(job_id, "replayed")
        return len(ids)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def list(self, status: Optional[str] = None, limit: int = 50) -> list[Job]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [Job.from_row(r) for r in rows]

    def events(self, job_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        with self._lock:
            if job_id:
                rows = self._conn.execute(
                    "SELECT * FROM events WHERE job_id = ? ORDER BY id", (job_id,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def pending(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status NOT IN ('succeeded', 'dead')"
            ).fetchone()[0]

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            c = self._conn
            counts = {s: 0 for s in ("queued", "running", "retrying", "succeeded", "dead")}
            for r in c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status"):
                counts[r["status"]] = r["n"]
            agg = c.execute(
                """SELECT COUNT(*) total,
                          COALESCE(SUM(attempts), 0) attempts,
                          COALESCE(SUM(cost_usd), 0) cost,
                          COALESCE(SUM(prompt_tokens), 0) pt,
                          COALESCE(SUM(completion_tokens), 0) ct,
                          COALESCE(SUM(replay_count), 0) replays
                   FROM jobs"""
            ).fetchone()
            lat = [r[0] for r in c.execute(
                "SELECT latency_ms FROM jobs WHERE status = 'succeeded' AND latency_ms IS NOT NULL "
                "ORDER BY latency_ms")]
            ev = {r["kind"]: r["n"] for r in c.execute(
                "SELECT kind, COUNT(*) n FROM events GROUP BY kind")}

        def pct(p: float) -> Optional[float]:
            if not lat:
                return None
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1)

        finished = counts["succeeded"] + counts["dead"]
        return {
            "counts": counts,
            "total_jobs": agg["total"],
            "success_rate": round(counts["succeeded"] / finished, 4) if finished else None,
            "attempts": agg["attempts"],
            "retries_scheduled": ev.get("retry_scheduled", 0),
            "leases_recovered": ev.get("lease_expired", 0),
            "stale_writes_rejected": ev.get("stale_write_rejected", 0),
            "duplicates_ignored": ev.get("duplicate_ignored", 0),
            "replays": agg["replays"],
            "latency_ms_p50": pct(0.50),
            "latency_ms_p95": pct(0.95),
            "prompt_tokens": agg["pt"],
            "completion_tokens": agg["ct"],
            "cost_usd": round(agg["cost"], 6),
        }
