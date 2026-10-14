"""Reliability checks for persistent, approval-gated remote execution."""
import hashlib
import json

import pytest

from kaggle_gpu_mcp.service import Service
from kaggle_gpu_mcp.worker import Worker
from kaggle_gpu_mcp.worker_lock import exclusive_worker_lock


class RecordingProvider:
    def __init__(self):
        self.pushed = []

    def push(self, folder, accelerator):
        self.pushed.append((folder, accelerator))
        return "Pushed"

    def status(self, ref):
        return "Kernel status: queued"

    def logs(self, ref):
        return ""


def test_tampered_approved_source_is_not_submitted(settings):
    provider = RecordingProvider()
    service = Service(settings, provider)
    job = service.submit("train.py", approved=True)
    package = settings.workspace / "packages" / job["job_id"] / "task.py"
    assert service.db.get(job["job_id"])["metadata"]["source_sha256"] == hashlib.sha256(
        package.read_bytes()
    ).hexdigest()
    package.write_text("print('unauthorized replacement')", encoding="utf-8")
    Worker(settings, provider).tick()
    assert service.db.get(job["job_id"])["status"] == "FAILED"
    assert provider.pushed == []


def test_second_worker_cannot_enter_same_workspace(settings):
    with exclusive_worker_lock(settings.workspace):
        with pytest.raises(RuntimeError, match="Another"):
            exclusive_worker_lock(settings.workspace).__enter__()


def test_no_queued_job_starvation_after_100_completed_records(settings):
    provider = RecordingProvider()
    service = Service(settings, provider)
    # Simulate more than 100 *terminal* records without generating large files.
    for index in range(105):
        record = service.db.create("custom", f"user/test-{index}", "dummy.py", "cpu", "debug", [], [])
        service.db.update(record["id"], "COMPLETED", "Historical job")
    job = service.submit("train.py", approved=True)
    Worker(settings, provider).tick()
    assert provider.pushed
    assert service.db.get(job["job_id"])["status"] in ("SUBMITTED", "QUEUED_REMOTE")


def test_terminal_records_cannot_be_reopened(settings):
    service = Service(settings)
    job = service.submit("train.py", approved=True)
    service.db.update(job["job_id"], "FAILED", "bad run")
    with pytest.raises(ValueError, match="Cannot transition"):
        service.db.update(job["job_id"], "SUBMITTED", "retry")
