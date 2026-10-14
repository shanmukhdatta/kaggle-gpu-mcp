"""Simple, explicit configuration; secrets never logged."""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    workspace: Path
    username: str
    kaggle_bin: str = "kaggle"
    poll_seconds: int = 45
    max_active_jobs: int = 1
    max_artifact_bytes: int = 512 * 1024 * 1024
    email_to: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_sender: str = ""
    smtp_ssl: bool = False
    allow_submissions: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        workspace = Path(os.getenv("KGM_WORKSPACE", "./workspace")).expanduser().resolve()
        return cls(
            workspace=workspace,
            username=os.getenv("KGM_KAGGLE_USERNAME", "").strip().lower(),
            kaggle_bin=os.getenv("KGM_KAGGLE_BIN", "kaggle"),
            poll_seconds=max(10, int(os.getenv("KGM_POLL_SECONDS", "45"))),
            max_active_jobs=max(1, int(os.getenv("KGM_MAX_ACTIVE_JOBS", "1"))),
            max_artifact_bytes=max(1024, int(os.getenv("KGM_MAX_ARTIFACT_BYTES", "536870912"))),
            email_to=os.getenv("KGM_NOTIFY_EMAIL", ""),
            smtp_host=os.getenv("KGM_SMTP_HOST", ""),
            smtp_port=int(os.getenv("KGM_SMTP_PORT", "587")),
            smtp_user=os.getenv("KGM_SMTP_USERNAME", ""),
            smtp_password=os.getenv("KGM_SMTP_PASSWORD", ""),
            smtp_sender=os.getenv("KGM_SMTP_SENDER", ""),
            smtp_ssl=os.getenv("KGM_SMTP_USE_SSL", "0") == "1",
            allow_submissions=os.getenv("KGM_ALLOW_SUBMISSIONS", "0") == "1",
        )

    def ensure(self) -> None:
        for name in ("packages", "artifacts", "logs", "datasets"):
            (self.workspace / name).mkdir(parents=True, exist_ok=True)
