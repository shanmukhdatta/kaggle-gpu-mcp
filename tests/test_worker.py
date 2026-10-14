import json
from pathlib import Path

from kaggle_gpu_mcp.config import Settings
from kaggle_gpu_mcp.service import Service
from kaggle_gpu_mcp.worker import Worker


class FakeKaggle:
    def __init__(self):
        self.pushed = []
        self.calls = {}

    def push(self, folder, accelerator):
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        self.pushed.append((meta, accelerator))
        return "Pushed"

    def status(self, ref):
        return "Kernel status: complete"

    def logs(self, ref):
        return "Training completed without errors"

    def output(self, ref, directory: Path):
        if ref.endswith(self.pushed[0][0]["id"].split("/")[-1]):
            (directory / "merged.csv").write_text("f1,f2,label\n0.1,0.2,1\n")
        else:
            (directory / "model.pt").write_bytes(b"test-statedict-placeholder")
            (directory / "metrics.json").write_text('{"status":"success","validation_accuracy":0.9}')
            (directory / "run_manifest.json").write_text('{"device":"cuda","gpu_verified":true}')
        return "Downloaded"


class FakeMail:
    def __init__(self):
        self.sent = []

    def enabled(self):
        return True

    def send(self, subject, message):
        self.sent.append((subject, message))
        return True


def test_two_stage_workflow_and_email(settings: Settings):
    p, mail = FakeKaggle(), FakeMail()
    svc = Service(settings, p)
    flow = svc.workflow("prep.py", "train.py", approved=True)
    w = Worker(settings, p, mail)
    for _ in range(5):
        w.tick()
    prep = svc.db.get(flow["preparation"]["job_id"])
    train = svc.db.get(flow["training"]["job_id"])
    assert prep["status"] == "COMPLETED"
    assert train["status"] == "COMPLETED"
    assert len(p.pushed) == 2
    assert p.pushed[0][1] is None
    assert p.pushed[1][1] == "NvidiaTeslaT4"
    assert prep["kernel_ref"] in p.pushed[1][0]["kernel_sources"]
    assert len(mail.sent) == 2
    assert svc.details(train["id"])["artifacts"]["run_manifest"]["gpu_verified"]
    Worker(settings, p, mail).tick()
    assert len(mail.sent) == 2  # No duplicate notifications across worker restarts


def test_failed_dependency(settings):
    class ErrorKaggle(FakeKaggle):
        def status(self, ref):
            return "Kernel status: error"
    p = ErrorKaggle()
    svc = Service(settings, p)
    flow = svc.workflow("prep.py", "train.py", approved=True)
    w = Worker(settings, p)
    w.tick()
    w.tick()
    child = svc.db.get(flow["training"]["job_id"])
    assert child["status"] == "FAILED"
    assert len(p.pushed) == 1


def test_unknown_submission_never_retried_automatically(settings):
    class UncertainKaggle(FakeKaggle):
        def push(self, folder, accelerator):
            self.pushed.append("called")
            raise TimeoutError("timeout after accept")
    p = UncertainKaggle()
    svc = Service(settings, p)
    job = svc.submit("prep.py", "cpu", "preprocess", approved=True, kind="prep")
    w = Worker(settings, p)
    w.tick()
    w.tick()
    assert svc.db.get(job["job_id"])["status"] == "SUBMISSION_UNKNOWN"
    assert len(p.pushed) == 1


def test_source_package_is_frozen_when_queued(settings):
    provider = FakeKaggle()
    service = Service(settings, provider)
    job = service.submit("train.py", approved=True)
    package = settings.workspace / "packages" / job["job_id"] / "task.py"
    original = package.read_bytes()
    (settings.workspace / "train.py").write_text("print('modified after approval')")
    Worker(settings, provider, FakeMail()).tick()
    assert package.read_bytes() == original
    assert provider.pushed  # The job was actually handed to the provider adapter.
