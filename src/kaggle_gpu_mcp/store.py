"""SQLite durable job and notification outbox; single-writer worker per workspace."""
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Terminal states never transition back into active work.
TERMINAL = frozenset({"COMPLETED", "FAILED"})


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.db() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                    kernel_ref TEXT NOT NULL UNIQUE, code_path TEXT NOT NULL,
                    mode TEXT NOT NULL, workload TEXT NOT NULL,
                    dataset_sources TEXT NOT NULL, kernel_sources TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    depends_on TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    last_error TEXT NOT NULL DEFAULT '', poll_after TEXT NOT NULL DEFAULT '',
                    notified INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    at TEXT NOT NULL, state TEXT NOT NULL, message TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL DEFAULT 'PENDING', tries INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS dataset_catalog (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    reference TEXT NOT NULL,
                    license_name TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dataset_catalog_registered
                    ON dataset_catalog(registered_at);
            """)
            # Safe migration for workspaces created with the initial schema.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(notifications)")}
            if "next_attempt_at" not in columns:
                conn.execute("ALTER TABLE notifications ADD COLUMN next_attempt_at TEXT NOT NULL DEFAULT ''")

    def db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, kind: str, kernel_ref: str, code_path: str, mode: str, workload: str,
               dataset_sources: list[str], kernel_sources: list[str], depends_on: str = "",
               metadata: dict | None = None) -> dict:
        job_id = uuid.uuid4().hex
        t = now()
        with self.db() as conn:
            conn.execute("""INSERT INTO jobs
                (id,kind,status,kernel_ref,code_path,mode,workload,dataset_sources,
                 kernel_sources,metadata,depends_on,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (job_id, kind, "WAITING" if depends_on else "QUEUED", kernel_ref, code_path,
                 mode, workload, json.dumps(dataset_sources), json.dumps(kernel_sources),
                 json.dumps(metadata or {}), depends_on or None, t, t))
            conn.execute("INSERT INTO events(job_id,at,state,message) VALUES(?,?,?,?)",
                         (job_id, t, "CREATED", "Job registered"))
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        with self.db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown job ID")
        data = dict(row)
        for key in ("dataset_sources", "kernel_sources", "metadata"):
            data[key] = json.loads(data[key])
        return data

    def list(self, limit: int = 25) -> list[dict]:
        with self.db() as conn:
            ids = [r[0] for r in conn.execute(
                "SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?", (min(max(1, limit), 100),))]
        return [self.get(k) for k in ids]

    def list_by_status(self, states: tuple[str, ...], limit: int = 10000) -> list[dict]:
        if not states:
            return []
        marks = ",".join("?" for _ in states)
        with self.db() as conn:
            ids = [r[0] for r in conn.execute(
                f"SELECT id FROM jobs WHERE status IN ({marks}) ORDER BY created_at ASC LIMIT ?",
                (*states, max(1, min(limit, 10000))))]
        return [self.get(k) for k in ids]

    def active_count(self) -> int:
        states = ("SUBMITTED", "RUNNING", "QUEUED_REMOTE", "SUBMITTING",
                  "SUBMISSION_UNKNOWN", "DOWNLOADING")
        marks = ",".join("?" for _ in states)
        with self.db() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE status IN ({marks})", states).fetchone()
        return int(row[0])

    def update(self, job_id: str, status: str, message: str = "", **fields: Any) -> None:
        allowed = {"last_error", "poll_after", "notified", "kernel_sources", "metadata"}
        if set(fields) - allowed:
            raise ValueError("Disallowed job field")
        if "kernel_sources" in fields:
            fields["kernel_sources"] = json.dumps(fields["kernel_sources"])
        if "metadata" in fields:
            fields["metadata"] = json.dumps(fields["metadata"])
        t = now()
        with self.db() as conn:
            previous = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if previous is None:
                raise KeyError("Unknown job ID")
            if previous[0] in TERMINAL and status != previous[0]:
                raise ValueError("Cannot transition a completed or failed job")
            conn.execute(
                "UPDATE jobs SET status=?,updated_at=?" +
                "".join(f",{k}=?" for k in fields) + " WHERE id=?",
                (status, t, *fields.values(), job_id))
            conn.execute("INSERT INTO events(job_id,at,state,message) VALUES(?,?,?,?)",
                         (job_id, t, status, message[:500]))
            if status in {"COMPLETED", "FAILED"}:
                conn.execute("INSERT OR IGNORE INTO notifications(job_id) VALUES(?)", (job_id,))

    def events(self, job_id: str) -> list[dict]:
        self.get(job_id)
        with self.db() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT at,state,message FROM events WHERE job_id=? ORDER BY id", (job_id,))]

    def notifications(self) -> list[dict]:
        with self.db() as conn:
            rows = conn.execute(
                """SELECT * FROM notifications WHERE state='PENDING' AND tries<3
                   AND (next_attempt_at='' OR next_attempt_at<=?) ORDER BY id LIMIT 10""",
                (now(),)).fetchall()
        return [dict(row) for row in rows]

    def list_notifications(self, limit: int = 50) -> list[dict]:
        with self.db() as conn:
            rows = conn.execute(
                "SELECT * FROM notifications ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 200)),)).fetchall()
        return [dict(row) for row in rows]

    def notification_update(self, row_id: int, state: str, error: str = "") -> None:
        """Retry with bounded backoff; stop after three failed deliveries."""
        with self.db() as conn:
            row = conn.execute("SELECT tries FROM notifications WHERE id=?", (row_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown notification")
            attempts = int(row[0]) + (1 if state in ("SENT", "PENDING") else 0)
            final = "EXHAUSTED" if state == "PENDING" and attempts >= 3 else state
            delayed = ""
            if final == "PENDING":
                delay = min(3600, 60 * (2 ** (attempts - 1)))
                delayed = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
            conn.execute("""UPDATE notifications SET state=?,tries=?,last_error=?,
                         next_attempt_at=? WHERE id=?""",
                         (final, attempts, error[:200], delayed, row_id))

    def retry_notification(self, job_id: str) -> dict:
        job = self.get(job_id)
        if job["status"] not in TERMINAL:
            raise ValueError("Only completed or failed jobs can send notifications")
        with self.db() as conn:
            row = conn.execute("SELECT * FROM notifications WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError("Terminal job has no notification record")
            if row["state"] == "SENT":
                raise ValueError("Already delivered; refusing duplicate email")
            conn.execute("""UPDATE notifications SET state='PENDING',tries=0,
                         last_error='',next_attempt_at='' WHERE job_id=?""", (job_id,))
        return {"job_id": job_id, "notification": "PENDING"}

    def register_dataset(self, source_type: str, reference: str, license_name: str,
                         metadata: dict) -> dict:
        """Record immutable dataset provenance. Does not upload dataset contents."""
        item = {"id": uuid.uuid4().hex, "source_type": source_type,
                "reference": reference, "license_name": license_name,
                "metadata": metadata, "registered_at": now()}
        with self.db() as conn:
            conn.execute("INSERT INTO dataset_catalog VALUES(?,?,?,?,?,?)", (
                item["id"], source_type, reference, license_name,
                json.dumps(metadata), item["registered_at"]))
        return item

    def datasets(self, limit: int = 50) -> list[dict]:
        with self.db() as conn:
            rows = conn.execute("SELECT * FROM dataset_catalog ORDER BY registered_at DESC LIMIT ?",
                                (max(1, min(limit, 200)),)).fetchall()
        return [dict(**{k: (json.loads(r[k]) if k == "metadata" else r[k])
                        for k in r.keys()}) for r in rows]
