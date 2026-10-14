"""SQLite durable job and notification outbox; single-writer worker per workspace."""
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
            """)

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
            return [dict(r) for r in conn.execute(
                "SELECT * FROM notifications WHERE state='PENDING' AND tries<3 ORDER BY id LIMIT 10")]

    def notification_update(self, row_id: int, state: str, error: str = "") -> None:
        with self.db() as conn:
            conn.execute("UPDATE notifications SET state=?, tries=tries+1,last_error=? WHERE id=?",
                         (state, error[:200], row_id))
