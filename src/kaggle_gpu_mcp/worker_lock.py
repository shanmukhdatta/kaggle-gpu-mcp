"""Cross-platform advisory worker lock held for the entire scheduler lifetime.

A separate worker process must not submit the same queued Kaggle job concurrently.
Unlike PID files, OS advisory locks are released automatically after a crash.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def exclusive_worker_lock(workspace: Path) -> Iterator[None]:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "worker.lock"
    with path.open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                handle.write(b" ")  # ensure the locked byte exists
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another Kaggle GPU MCP worker is active for this workspace") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
