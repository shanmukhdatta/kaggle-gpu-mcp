from pathlib import Path

import pytest

from kaggle_gpu_mcp.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(workspace=tmp_path / "workspace", username="testuser", poll_seconds=10, allow_submissions=True)
    s.ensure()
    (s.workspace / "prep.py").write_text("print('prep')\n", encoding="utf-8")
    (s.workspace / "train.py").write_text("print('train')\n", encoding="utf-8")
    return s
