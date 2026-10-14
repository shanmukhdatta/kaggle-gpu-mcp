import json

import pytest

from kaggle_gpu_mcp.artifacts import ArtifactError, manifest
from kaggle_gpu_mcp.demo import install_demo
from kaggle_gpu_mcp.provider import interpret_status
from kaggle_gpu_mcp.resources import choose
from kaggle_gpu_mcp.security import allow_file, validate_ref
from kaggle_gpu_mcp.service import Service


def test_resource_selection():
    assert choose("auto", "pytorch")["selected"] == "gpu"
    assert choose("auto", "preprocess")["selected"] == "cpu"
    assert choose("cpu", "pytorch")["selected"] == "cpu"
    with pytest.raises(ValueError):
        choose("tpu", "pytorch")


@pytest.mark.parametrize("text,expected", [
    ("Kernel status: complete", "SUCCEEDED_REMOTE"),
    ("Kernel status: error", "FAILED"),
    ("Kernel status: running", "RUNNING"),
    ("Kernel status: queued", "QUEUED_REMOTE"),
    ("HTTP error unknown", "UNKNOWN_REMOTE"),
])
def test_status(text, expected):
    assert interpret_status(text) == expected


def test_workspace_guard(settings, tmp_path):
    with pytest.raises(ValueError):
        allow_file(settings.workspace, str(tmp_path / "outside.py"), (".py",))
    with pytest.raises(ValueError):
        validate_ref("--injected")


def test_approval_and_submission(settings):
    svc = Service(settings)
    with pytest.raises(ValueError):
        svc.submit("train.py")
    job = svc.submit("train.py", mode="auto", approved=True)
    assert job["selected_resource"]["selected"] == "gpu"
    record = svc.db.get(job["job_id"])
    package = settings.workspace / "packages" / job["job_id"]
    meta = json.loads((package / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
    assert meta["id"] == record["kernel_ref"]
    assert "kernel_sources" in meta
    assert (package / "task.py").exists()


def test_workflow_dependencies(settings):
    svc = Service(settings)
    flow = svc.workflow("prep.py", "train.py", approved=True)
    parent = svc.db.get(flow["preparation"]["job_id"])
    child = svc.db.get(flow["training"]["job_id"])
    assert parent["status"] == "QUEUED"
    assert child["status"] == "WAITING"
    assert child["depends_on"] == parent["id"]


def test_artifact_verification(settings):
    root = settings.workspace / "artifacts" / "verify"
    root.mkdir(parents=True)
    (root / "model.pt").write_bytes(b"stand-in")
    (root / "metrics.json").write_text('{"status":"success"}')
    (root / "run_manifest.json").write_text('{"gpu_verified":true,"device":"cuda"}')
    report = manifest(root, 10_000, True, True)
    assert report["run_manifest"]["gpu_verified"] is True
    assert len(report["files"]) == 3
    (root / "run_manifest.json").write_text('{"gpu_verified":false,"device":"cpu"}')
    with pytest.raises(ArtifactError):
        manifest(root, 10_000, True, True)


def test_demo_install_and_no_overwrite(settings):
    result = install_demo(settings)
    assert (settings.workspace / result["train_code"]).is_file()
    with pytest.raises(ValueError):
        install_demo(settings)


def test_merge_local_csv(settings):
    svc = Service(settings)
    (settings.workspace / "a.csv").write_text("x,y\n1,2\n2,3\n")
    (settings.workspace / "b.csv").write_text("x,y\n1,2\n3,4\n")
    result = svc.merge_csv(["a.csv", "b.csv"])
    assert result["rows"] == 3
    assert (settings.workspace / result["relative_path"]).exists()
    with pytest.raises(ValueError):
        svc.merge_csv(["a.csv", "b.csv"])


def test_followup_and_compare(settings):
    svc = Service(settings)
    a = svc.submit("train.py", approved=True)
    with pytest.raises(ValueError):
        svc.followup(a["job_id"], "train.py", approved=True)
    with pytest.raises(ValueError):
        svc.compare([a["job_id"], a["job_id"]])
