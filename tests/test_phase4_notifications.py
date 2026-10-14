"""Durable notification inbox and retry safety."""
import pytest

from kaggle_gpu_mcp.service import Service
from kaggle_gpu_mcp.worker import Worker


class Provider:
    def __init__(self):
        self.pushed = 0

    def push(self, folder, accelerator):
        self.pushed += 1
        return "submitted"

    def status(self, ref):
        return "Kernel status: error"

    def logs(self, ref):
        return "Remote execution failed"


class FlakyMailer:
    def __init__(self):
        self.attempts = 0

    def enabled(self):
        return True

    def send(self, subject, message):
        self.attempts += 1
        raise ConnectionError("SMTP unavailable")


def test_backoff_and_manual_retry(settings):
    provider, mail = Provider(), FlakyMailer()
    service = Service(settings, provider)
    job = service.submit("train.py", approved=True)
    worker = Worker(settings, provider, mail)
    worker.tick()
    assert service.db.get(job["job_id"])["status"] == "FAILED"
    notifications = service.db.list_notifications()
    assert notifications[0]["tries"] == 1
    assert notifications[0]["next_attempt_at"]
    # Retry is not attempted again on the next tick.
    worker.tick()
    assert mail.attempts == 1
    with pytest.raises(ValueError):
        service.notification_retry(job["job_id"])
    assert service.notification_retry(job["job_id"], approved=True)["notification"] == "PENDING"
    worker.tick()
    assert mail.attempts == 2


def test_inbox_no_duplicate_after_sent(settings):
    class SuccessMailer(FlakyMailer):
        def send(self, subject, message):
            self.attempts += 1
            return True
    mail = SuccessMailer()
    provider = Provider()
    service = Service(settings, provider)
    job = service.submit("train.py", approved=True)
    worker = Worker(settings, provider, mail)
    worker.tick()
    assert service.db.list_notifications()[0]["state"] == "SENT"
    with pytest.raises(ValueError, match="Already delivered"):
        service.notification_retry(job["job_id"], approved=True)
    worker.tick()
    assert mail.attempts == 1


def test_schema_migration(settings):
    import sqlite3
    old = settings.workspace / "old.sqlite"
    with sqlite3.connect(old) as conn:
        conn.execute("""CREATE TABLE notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL UNIQUE,
            state TEXT NOT NULL DEFAULT 'PENDING', tries INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '')""")
    from kaggle_gpu_mcp.store import Store
    db = Store(old)
    with db.db() as conn:
        names = [x[1] for x in conn.execute("PRAGMA table_info(notifications)")]
    assert "next_attempt_at" in names


def test_workflow_status_pending(settings):
    service = Service(settings)
    flow = service.workflow("prep.py", "train.py", approved=True)
    result = service.workflow_status(flow["training"]["job_id"])
    assert result["status"] == "RUNNING"
    assert result["preparation"]["status"] == "QUEUED"
