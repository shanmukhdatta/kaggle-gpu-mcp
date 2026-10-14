"""Validated tool-use operations and immutable job packaging."""
import hashlib
import json
import re
import shutil
from pathlib import Path

from .config import Settings
from .dataset_catalog import profile_csv, sha256_file
from .provider import KaggleCLI, interpret_status
from .resources import choose
from .security import allow_dir, allow_file, validate_ref, validate_slug, validate_sources, validate_username
from .store import Store


class Service:
    def __init__(self, settings: Settings, provider: KaggleCLI | None = None):
        self.s = settings
        settings.ensure()
        self.db = Store(settings.workspace / "jobs.sqlite3")
        self.provider = provider or KaggleCLI(settings)

    def select_resource(self, mode: str, workload: str = "pytorch") -> dict:
        return choose(mode, workload)

    def submit(self, code_path: str, mode: str = "auto", workload: str = "pytorch",
               dataset_sources: list[str] | None = None, kernel_sources: list[str] | None = None,
               kind: str = "train", approved: bool = False, depends_on: str = "",
               require_contract: bool = True) -> dict:
        if not approved or not self.s.allow_submissions:
            raise ValueError("Submission requires approved=true AND host-set KGM_ALLOW_SUBMISSIONS=1")
        validate_username(self.s.username)
        if kind not in {"train", "prep", "custom"}:
            raise ValueError("kind must be train, prep, or custom")
        choice = choose(mode, workload)
        code = allow_file(self.s.workspace, code_path, (".py", ".ipynb"))
        datasets = validate_sources(dataset_sources or [])
        kernels = validate_sources(kernel_sources or [])
        if depends_on:
            parent = self.db.get(depends_on)
            if parent["status"] in {"FAILED"}:
                raise ValueError("Cannot depend on a failed job")
            # Dependent source is added by worker after parent completion.
        # Each job gets a unique slug, avoiding latest-version races.
        # A generated UUID is used instead of user-controlled metadata.
        unique = __import__("uuid").uuid4().hex[:16]
        slug = validate_slug("kgm-" + unique)
        ref = f"{self.s.username}/{slug}"
        job = self.db.create(kind, ref, str(code), choice["selected"], workload,
                             datasets, kernels, depends_on=depends_on,
                             metadata={"require_contract": require_contract,
                                       "requested_mode": mode})
        self._package(job)
        return {"job_id": job["id"], "status": job["status"],
                "kernel_ref": ref, "selected_resource": choice, "depends_on": depends_on}

    def _package(self, job: dict) -> Path:
        folder = self.s.workspace / "packages" / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        code = Path(job["code_path"])
        file_name = "task" + code.suffix.lower()
        shutil.copy2(code, folder / file_name)
        # Hash the approved bytes. Later edits to the workspace cannot silently
        # modify the approved remote execution package.
        code_hash = hashlib.sha256((folder / file_name).read_bytes()).hexdigest()
        job_meta = dict(job["metadata"])
        job_meta["source_sha256"] = code_hash
        self.db.update(job["id"], job["status"], "Approved package frozen", metadata=job_meta)
        slug = job["kernel_ref"].split("/")[1]
        # Title and slug are intentionally identical: Kaggle ties notebook titles to slugs.
        metadata = {
            "id": job["kernel_ref"], "title": slug,
            "code_file": file_name, "language": "python",
            "kernel_type": "notebook" if code.suffix.lower() == ".ipynb" else "script",
            "is_private": True, "enable_gpu": job["mode"] == "gpu",
            "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4" if job["mode"] == "gpu" else "",
            "dataset_sources": job["dataset_sources"],
            "competition_sources": [], "kernel_sources": job["kernel_sources"], "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        return folder

    def details(self, job_id: str) -> dict:
        j = self.db.get(job_id)
        result = {k: v for k, v in j.items() if k != "code_path"}
        result["events"] = self.db.events(job_id)
        summary = self.s.workspace / "artifacts" / job_id / "manifest.json"
        if summary.is_file():
            result["artifacts"] = json.loads(summary.read_text(encoding="utf-8"))
        return result

    def logs(self, job_id: str, max_chars: int = 12000) -> str:
        self.db.get(job_id)
        log_path = self.s.workspace / "logs" / f"{job_id}.txt"
        if not log_path.exists():
            return "No logs downloaded yet"
        max_chars = min(max(max_chars, 1), 50000)
        return log_path.read_text(encoding="utf-8", errors="replace")[-max_chars:]

    def publish_dataset(self, directory: str, slug: str, license_name: str,
                        version_note: str = "", approved: bool = False) -> dict:
        if not approved or not self.s.allow_submissions:
            raise ValueError("Publishing requires approved=true AND host-set KGM_ALLOW_SUBMISSIONS=1")
        validate_username(self.s.username)
        slug = validate_slug(slug)
        source = allow_dir(self.s.workspace, directory)
        files = sorted(p for p in source.iterdir() if p.is_file() and p.name != "dataset-metadata.json")
        if not files:
            raise ValueError("Dataset directory is empty")
        if any(p.is_symlink() for p in files):
            raise ValueError("Symlink sources are not supported")
        if any(p.suffix.lower() not in {".csv", ".json", ".parquet", ".txt", ".zip"} for p in files):
            raise ValueError("Unsupported dataset file type")
        if sum(p.stat().st_size for p in files) > self.s.max_artifact_bytes:
            raise ValueError("Dataset exceeds configured local upload limit")
        if not re.fullmatch(r"[A-Za-z0-9.-]{2,40}", license_name):
            raise ValueError("Invalid license identifier")
        # Never mutate an input folder; build a dedicated sanitized upload directory.
        dst = self.s.workspace / "datasets" / "_uploads" / slug
        dst.mkdir(parents=True, exist_ok=True)
        for existing in dst.iterdir():
            if existing.is_file():
                existing.unlink()
        for file in files:
            shutil.copy2(file, dst / file.name)
        ref = f"{self.s.username}/{slug}"
        (dst / "dataset-metadata.json").write_text(json.dumps({
            "title": (slug.replace("-", " ").title() + " Dataset")[:50],
            "id": ref,
            "licenses": [{"name": license_name}],
        }, indent=2), encoding="utf-8")
        output = (self.provider.dataset_version(dst, version_note)
                  if version_note else self.provider.dataset_create(dst))
        return {"dataset_ref": ref, "result": output[:800], "private": True}

    def workflow(self, prep_code: str, train_code: str, dataset_sources: list[str] | None = None,
                 approved: bool = False) -> dict:
        """Two Kaggle jobs: CPU preparation then GPU training from prep notebook outputs."""
        if not approved or not self.s.allow_submissions:
            raise ValueError("Workflow requires approved=true AND host-set KGM_ALLOW_SUBMISSIONS=1")
        # Validate both first, to avoid orphan preparation job.
        allow_file(self.s.workspace, prep_code, (".py", ".ipynb"))
        allow_file(self.s.workspace, train_code, (".py", ".ipynb"))
        sources = validate_sources(dataset_sources or [])
        prep = self.submit(prep_code, "cpu", "preprocess", sources, [], "prep", True,
                           require_contract=False)
        train = self.submit(train_code, "gpu", "pytorch", [], [], "train", True,
                            depends_on=prep["job_id"])
        return {"preparation": prep, "training": train, "note":
                "Training begins after prep output is saved and is attached as a Kaggle kernel source"}

    def reconcile(self, job_id: str) -> dict:
        """Conservatively reconcile an upload whose outcome was unknown after timeout."""
        job = self.db.get(job_id)
        if job["status"] != "SUBMISSION_UNKNOWN":
            raise ValueError("Only unknown submissions can be reconciled")
        try:
            state = interpret_status(self.provider.status(job["kernel_ref"]))
        except Exception:
            return {"job_id": job_id, "status": "SUBMISSION_UNKNOWN",
                    "action": "Check the Kaggle notebook URL before resubmitting"}
        if state == "UNKNOWN_REMOTE":
            return {"job_id": job_id, "status": "SUBMISSION_UNKNOWN",
                    "action": "Provider response unrecognized; no duplicate upload attempted"}
        target = "DOWNLOADING" if state == "SUCCEEDED_REMOTE" else (
            "FAILED" if state == "FAILED" else "SUBMITTED")
        self.db.update(job_id, target, "Manually reconciled Kaggle submission")
        return {"job_id": job_id, "status": target, "remote_state": state}

    def refresh_logs(self, job_id: str) -> str:
        """On-demand fetching of remote logs (may not be live)."""
        job = self.db.get(job_id)
        if job["status"] in {"QUEUED", "WAITING", "SUBMITTING"}:
            return "Job has not been confirmed on Kaggle yet"
        raw = self.provider.logs(job["kernel_ref"])
        log_path = self.s.workspace / "logs" / f"{job_id}.txt"
        log_path.write_text(raw[:200_000], encoding="utf-8")
        return self.logs(job_id)

    def compare(self, job_ids: list[str]) -> dict:
        """Return consistent metric reports for side-by-side LLM comparison."""
        if not (2 <= len(job_ids) <= 20):
            raise ValueError("Compare between 2 and 20 jobs")
        out = []
        for job_id in job_ids:
            job = self.db.get(job_id)
            if job["status"] != "COMPLETED":
                raise ValueError(f"Job {job_id} is not completed")
            report_file = self.s.workspace / "artifacts" / job_id / "manifest.json"
            if not report_file.exists():
                raise ValueError(f"Missing manifest for {job_id}")
            report = json.loads(report_file.read_text(encoding="utf-8"))
            out.append({"job_id": job_id, "kernel_ref": job["kernel_ref"],
                        "metrics": report["metrics"], "device": report["run_manifest"]})
        return {"experiments": out, "note": "Do not compare metrics from different splits or datasets as equivalent"}

    def followup(self, previous_job_id: str, code_path: str, approved: bool = False,
                 mode: str = "auto", workload: str = "pytorch") -> dict:
        """Submit a fresh code revision with the prior Kaggle output attached as input."""
        if not approved or not self.s.allow_submissions:
            raise ValueError("Follow-up requires approved=true AND host-set KGM_ALLOW_SUBMISSIONS=1")
        previous = self.db.get(previous_job_id)
        if previous["status"] != "COMPLETED":
            raise ValueError("Previous experiment is not complete")
        return self.submit(code_path, mode, workload, previous["dataset_sources"],
                           [*previous["kernel_sources"], previous["kernel_ref"]],
                           "train", True, require_contract=True)

    def register_source(self, source_type: str, reference: str, license_name: str,
                        approved: bool = False) -> dict:
        """Register a Kaggle reference or an approved workspace CSV; do not download."""
        if not approved:
            raise ValueError("Dataset provenance registration requires approved=true")
        if not re.fullmatch(r"[A-Za-z0-9.-]{2,40}", license_name):
            raise ValueError("License identifier is required")
        if source_type == "kaggle":
            clean = validate_ref(reference)
            metadata = {"discovery": "user-supplied-reference", "checked_remote": False}
        elif source_type == "local_csv":
            file = allow_file(self.s.workspace, reference, (".csv",))
            clean = file.relative_to(self.s.workspace).as_posix()
            metadata = profile_csv(file)
        else:
            raise ValueError("source_type must be kaggle or local_csv")
        return self.db.register_dataset(source_type, clean, license_name, metadata)

    def csv_profile(self, source_path: str) -> dict:
        """Report schema and missing data without returning private row samples."""
        return profile_csv(allow_file(self.s.workspace, source_path, (".csv",)))

    def workflow_status(self, training_job_id: str) -> dict:
        """Report the durable dependency chain and current saved artifacts."""
        training = self.db.get(training_job_id)
        parent_id = training["depends_on"]
        if not parent_id:
            raise ValueError("Job is not part of a dependent workflow")
        parent = self.db.get(parent_id)
        child_details = self.details(training_job_id)
        status = "FAILED" if "FAILED" in (parent["status"], training["status"]) else (
            "COMPLETED" if training["status"] == "COMPLETED" else "RUNNING")
        return {
            "status": status,
            "preparation": {"job_id": parent_id, "status": parent["status"],
                            "kernel_ref": parent["kernel_ref"]},
            "training": {"job_id": training_job_id, "status": training["status"],
                         "kernel_ref": training["kernel_ref"]},
            "artifacts": child_details.get("artifacts", {"status": "not_downloaded"}),
        }

    def notification_retry(self, job_id: str, approved: bool = False) -> dict:
        if not approved:
            raise ValueError("Retrying notification requires approved=true")
        return self.db.retry_notification(job_id)

    def merge_csv(self, paths: list[str], output_name: str = "combined.csv") -> dict:
        """Merge same-schema local CSV files. Explicit files only; no web scraping."""
        import csv
        if not (1 <= len(paths) <= 30):
            raise ValueError("Specify 1–30 CSV files")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}\.csv", output_name):
            raise ValueError("Output must be a plain .csv file name")
        source_paths = [allow_file(self.s.workspace, p, (".csv",)) for p in paths]
        # Reject mismatched schema BEFORE writing output; safe on Windows too.
        schemas = []
        for p in source_paths:
            with p.open(newline="", encoding="utf-8-sig") as handle:
                schemas.append(next(csv.reader(handle), []))
        if not schemas[0] or any(fields != schemas[0] for fields in schemas):
            raise ValueError("Cannot combine CSV files with missing or mismatched headers")
        # refuse excessively large workloads on the MCP server
        if sum(p.stat().st_size for p in source_paths) > self.s.max_artifact_bytes:
            raise ValueError("Dataset exceeds local size limit; use Kaggle CPU prep")
        output_folder = self.s.workspace / "datasets" / "merged"
        output_folder.mkdir(parents=True, exist_ok=True)
        target = output_folder / output_name
        if target.exists() or any(p.resolve() == target.resolve() for p in source_paths):
            raise ValueError("Output exists or would overwrite input")
        import os
        import tempfile
        schema = None
        seen = set()
        count = 0
        fd, temp_name = tempfile.mkstemp(prefix=".merge-", suffix=".csv", dir=output_folder)
        os.close(fd)
        temporary = Path(temp_name)
        try:
            with temporary.open("w", newline="", encoding="utf-8") as dest:
                writer = None
                for path in source_paths:
                    with path.open(newline="", encoding="utf-8-sig") as source:
                        reader = csv.DictReader(source)
                        if not reader.fieldnames:
                            raise ValueError("CSV missing header")
                        if schema is None:
                            schema = reader.fieldnames
                            if len(set(schema)) != len(schema):
                                raise ValueError("Duplicate input CSV headers")
                            writer = csv.DictWriter(dest, fieldnames=schema)
                            writer.writeheader()
                        if reader.fieldnames != schema:
                            raise ValueError("Cannot combine CSV files with mismatched schemas")
                        for row in reader:
                            if None in row:
                                raise ValueError("CSV row has more fields than header")
                            key = tuple(row.get(k, "") for k in schema)
                            if key not in seen:
                                seen.add(key)
                                writer.writerow(row)
                                count += 1
            if temporary.stat().st_size > self.s.max_artifact_bytes:
                raise ValueError("Merged CSV exceeds configured artifact limit")
            provenance = {
                "input_sources": [{"path": p.relative_to(self.s.workspace).as_posix(),
                                   "sha256": sha256_file(p)} for p in source_paths],
                "output_file": output_name,
                "output_sha256": sha256_file(temporary),
                "columns": schema, "rows": count,
                "note": "Exact-row deduplication; not leakage-safe dataset splitting",
            }
            # The sidecar is named by output basename so multiple merges remain distinct.
            sidecar = output_folder / f"{output_name}.manifest.json"
            if sidecar.exists() or target.exists():
                raise ValueError("Output or provenance manifest already exists")
            temporary.replace(target)
            sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        finally:
            temporary.unlink(missing_ok=True)
        return {"file": str(target), "relative_path": "datasets/merged/"+output_name,
                "rows": count, "columns": schema, "source_count": len(paths),
                "provenance_manifest": "datasets/merged/" + output_name + ".manifest.json",
                "sha256": provenance["output_sha256"]}
