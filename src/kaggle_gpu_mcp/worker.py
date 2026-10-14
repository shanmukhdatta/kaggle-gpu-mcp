"""Single-instance worker: queued submission, remote polling, outputs, notification outbox."""
import hashlib
import json
import shutil
import time
from datetime import datetime, timedelta, timezone

from .artifacts import manifest
from .config import Settings
from .notify import Mailer
from .provider import KaggleCLI, interpret_status
from .service import Service
from .store import now
from .worker_lock import exclusive_worker_lock


class Worker:
    def __init__(self, settings: Settings, provider: KaggleCLI | None = None,
                 mailer: Mailer | None = None):
        self.service = Service(settings, provider=provider)
        self.s = settings
        self.p = self.service.provider
        self.mail = mailer or Mailer(settings)

    def _error(self, job: dict, exc: Exception, prefix: str) -> None:
        # Avoid embedding exception strings from Kaggle/SMTP (can leak secrets).
        self.service.db.update(job["id"], "FAILED", prefix,
                               last_error=f"{prefix}: {type(exc).__name__}")

    def _submit(self, job: dict) -> None:
        # The package was frozen when the job was queued; never recopy editable source.
        folder = self.s.workspace / "packages" / job["id"]
        metadata_path = folder / "kernel-metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            code_path = folder / metadata["code_file"]
            expected_hash = job["metadata"]["source_sha256"]
            if code_path.name not in {"task.py", "task.ipynb"} or not code_path.is_file():
                raise ValueError("Missing frozen training source")
            actual_hash = hashlib.sha256(code_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError("Approved source code modified after job registration")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self._error(job, exc, "Approved package integrity check failed")
            return
        # Dependency references become known only after the preparation job finishes.
        # Update generated metadata while preserving the frozen approved source code.
        if job["depends_on"]:
            metadata["kernel_sources"] = job["kernel_sources"]
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # IMPORTANT: distinguish unknown after push timeout from a definite failure.
        self.service.db.update(job["id"], "SUBMITTING", "Starting Kaggle push")
        try:
            self.p.push(folder, "NvidiaTeslaT4" if job["mode"] == "gpu" else None)
        except Exception:
            # Do not blindly re-submit: Kaggle may have accepted an upload before timeout.
            self.service.db.update(job["id"], "SUBMISSION_UNKNOWN",
                                   "Submission outcome unknown: reconcile manually using Kaggle reference")
            return
        self.service.db.update(job["id"], "SUBMITTED", "Notebook pushed to Kaggle")

    def _poll(self, job: dict) -> None:
        if job["poll_after"] and job["poll_after"] > now():
            return
        try:
            raw = self.p.status(job["kernel_ref"])
        except Exception:
            self._delay(job, "Status check temporarily unavailable")
            return
        state = interpret_status(raw)
        if state == "SUCCEEDED_REMOTE":
            self.service.db.update(job["id"], "DOWNLOADING", "Remote run succeeded")
            self._download(self.service.db.get(job["id"]))
        elif state == "FAILED":
            self.service.db.update(job["id"], "FAILED", "Kaggle reports remote failure",
                                   last_error="Remote Kaggle run failed")
            self._fetch_logs(job)
        elif state in {"QUEUED_REMOTE", "RUNNING"}:
            self._fetch_logs(job)
            self._delay(job, state, state)
        else:
            self._delay(job, "Unrecognized Kaggle status; retrying")

    def _delay(self, job: dict, description: str, state: str = "SUBMITTED") -> None:
        next_at = (datetime.now(timezone.utc) + timedelta(seconds=self.s.poll_seconds)).isoformat()
        self.service.db.update(job["id"], state, description, poll_after=next_at)

    def _fetch_logs(self, job: dict) -> None:
        try:
            log = self.p.logs(job["kernel_ref"])
        except Exception:
            return
        # stderr from tools is never displayed; log file may still include notebook secrets.
        path = self.s.workspace / "logs" / f"{job['id']}.txt"
        path.write_text(log[:200_000], encoding="utf-8")

    def _download(self, job: dict) -> None:
        folder = self.s.workspace / "artifacts" / job["id"]
        # Avoid accepting stale partial files from an interrupted download attempt.
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            self.p.output(job["kernel_ref"], folder)
            report = manifest(folder, self.s.max_artifact_bytes,
                              bool(job["metadata"].get("require_contract", True)),
                              job["mode"] == "gpu" and job["kind"] == "train")
            (folder / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            self._fetch_logs(job)
        except Exception as exc:
            self._error(job, exc, "Download or artifact verification failed")
            return
        self.service.db.update(job["id"], "COMPLETED", "Artifacts verified and stored")
        for child in self.service.db.list(limit=100):
            if child["depends_on"] == job["id"] and child["status"] == "WAITING":
                kernels = list(child["kernel_sources"]) + [job["kernel_ref"]]
                self.service.db.update(child["id"], "QUEUED", "Dependency completed",
                                       kernel_sources=kernels)

    def _notify(self) -> None:
        for n in self.service.db.notifications():
            job = self.service.db.get(n["job_id"])
            if not self.mail.enabled():
                self.service.db.notification_update(n["id"], "SKIPPED", "Email not configured")
                continue
            try:
                sent = self.mail.send(
                    f"Kaggle MCP: {job['status']} — {job['id'][:12]}",
                    f"Job: {job['id']}\nStatus: {job['status']}\nKaggle: {job['kernel_ref']}\n"
                    f"Local artifacts: {self.s.workspace / 'artifacts' / job['id']}\n"
                    f"Error: {job['last_error']}\n")
                self.service.db.notification_update(n["id"], "SENT" if sent else "SKIPPED")
            except Exception:
                self.service.db.notification_update(n["id"], "PENDING", "SMTP delivery failed")

    def tick(self) -> None:
        with exclusive_worker_lock(self.s.workspace):
            self._tick_unlocked()

    def _tick_unlocked(self) -> None:
        # SQL status queries prevent starvation once more than 100 jobs exist.
        for job in self.service.db.list_by_status(("WAITING",)):
            if job["status"] == "WAITING" and job["depends_on"]:
                if self.service.db.get(job["depends_on"])["status"] == "FAILED":
                    self.service.db.update(job["id"], "FAILED", "Dependency failed",
                                           last_error="Preparation job failed")
        for job in self.service.db.list_by_status(("SUBMITTING", "DOWNLOADING")):
            if job["status"] == "SUBMITTING":
                self.service.db.update(job["id"], "SUBMISSION_UNKNOWN",
                                       "Worker restarted during upload; manual reconciliation required")
            if job["status"] == "DOWNLOADING":
                self._download(job)
        # No more than configured simultaneous provider jobs.
        active = self.service.db.active_count()
        if active < self.s.max_active_jobs:
            for job in self.service.db.list_by_status(("QUEUED",), limit=1):
                self._submit(job)
        for job in self.service.db.list_by_status(("SUBMITTED", "RUNNING", "QUEUED_REMOTE")):
            self._poll(job)
        self._notify()

    def run_forever(self) -> None:
        print("Kaggle GPU MCP worker running; stop with Ctrl+C", flush=True)
        with exclusive_worker_lock(self.s.workspace):
            while True:
                self._tick_unlocked()
                time.sleep(self.s.poll_seconds)
