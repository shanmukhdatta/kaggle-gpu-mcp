"""Defensive downloaded-artifact inspection; no pickle deserialization."""
import hashlib
import json
from pathlib import Path


class ArtifactError(ValueError):
    pass


def manifest(root: Path, max_bytes: int, require_contract: bool, require_gpu: bool) -> dict:
    files: list[dict] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ArtifactError("Symlinks in Kaggle output are not allowed")
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ArtifactError("Downloaded artifact escapes its output directory")
        size = path.stat().st_size
        total += size
        if total > max_bytes:
            raise ArtifactError("Downloaded artifacts exceed configured size limit")
        h = hashlib.sha256()
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
        files.append({"name": path.relative_to(root).as_posix(), "bytes": size, "sha256": h.hexdigest()})
    if not files:
        raise ArtifactError("Kaggle job produced no downloadable files")
    metrics = {}
    device = {}
    if require_contract:
        metrics_file = root / "metrics.json"
        device_file = root / "run_manifest.json"
        if not metrics_file.is_file() or not device_file.is_file():
            raise ArtifactError("Missing metrics.json or run_manifest.json")
        if metrics_file.stat().st_size > 1024 * 1024 or device_file.stat().st_size > 1024 * 1024:
            raise ArtifactError("Manifest/metrics too large")
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        device = json.loads(device_file.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict) or metrics.get("status") != "success":
            raise ArtifactError("Training metrics do not report success")
        if require_gpu and not (device.get("gpu_verified") is True and device.get("device") == "cuda"):
            raise ArtifactError("Actual GPU computation was not verified")
        if not (root / "model.pt").exists() and require_gpu:
            raise ArtifactError("GPU training did not produce model.pt")
    return {"files": files, "total_bytes": total, "metrics": metrics, "run_manifest": device}
