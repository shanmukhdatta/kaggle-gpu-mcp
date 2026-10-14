"""Kaggle CLI adapter, no shell invocation. Do not expose arbitrary command execution."""
import os
import re
import subprocess
from pathlib import Path

from .config import Settings
from .security import validate_ref


class KaggleError(RuntimeError):
    pass


class KaggleCLI:
    def __init__(self, settings: Settings):
        self.s = settings

    def _run(self, args: list[str], timeout: int = 120) -> str:
        # A Kaggle credentials file / environment is handled by Kaggle itself.
        env = os.environ.copy()
        try:
            result = subprocess.run(
                [self.s.kaggle_bin, *args], capture_output=True, text=True,
                timeout=timeout, env=env, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise KaggleError(f"Kaggle command unavailable or timed out: {type(exc).__name__}") from exc
        if result.returncode:
            # Don't return potentially sensitive Kaggle stderr to an AI client.
            operation = " ".join(args[:2])
            raise KaggleError(
                f"Kaggle {operation} failed (exit {result.returncode}); check worker logs"
            )
        return result.stdout[:100_000]

    def check(self) -> dict:
        version = self._run(["--version"], 20)
        mine = self._run(["kernels", "list", "-m"], 40)
        return {"cli_version": version.strip(), "authenticated": True, "list_sample": mine[:200]}

    def push(self, folder: Path, accelerator: str | None) -> str:
        args = ["kernels", "push", "-p", str(folder)]
        if accelerator:
            args += ["--accelerator", accelerator]
        return self._run(args, 900)

    def status(self, ref: str) -> str:
        return self._run(["kernels", "status", validate_ref(ref)], 60)

    def logs(self, ref: str) -> str:
        return self._run(["kernels", "logs", validate_ref(ref)], 90)

    def output(self, ref: str, directory: Path) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        return self._run(["kernels", "output", validate_ref(ref), "-p", str(directory)], 1200)

    def dataset_create(self, folder: Path) -> str:
        return self._run(["datasets", "create", "-p", str(folder), "-t"], 1200)

    def dataset_version(self, folder: Path, note: str) -> str:
        if not note or len(note) > 200:
            raise ValueError("Provide version notes (1–200 characters)")
        return self._run(["datasets", "version", "-p", str(folder), "-m", note, "-t"], 1200)


def interpret_status(output: str) -> str:
    """Parse known Kaggle CLI states conservatively; unknown never implies completion."""
    s = output.lower()
    # Example: 'Kernel status: complete' / 'Kernel status: error'
    match = re.search(r"(?:kernel\s+)?status\s*:\s*([\w-]+)", s)
    state = match.group(1) if match else s.strip()
    if state in {"complete", "completed", "success", "finished"}:
        return "SUCCEEDED_REMOTE"
    if state in {"error", "failed", "failure", "cancelled", "canceled"}:
        return "FAILED"
    if state in {"running", "executing"}:
        return "RUNNING"
    if state in {"pending", "queued", "initializing"}:
        return "QUEUED_REMOTE"
    return "UNKNOWN_REMOTE"
