"""Generate editable demo scripts in configured workspace."""
import shutil
from importlib.resources import files
from pathlib import Path

from .config import Settings


def install_demo(settings: Settings) -> dict:
    dest = settings.workspace / "demo"
    dest.mkdir(parents=True, exist_ok=True)
    for part in ("prep", "train"):
        target = dest / f"{part}.py"
        if target.exists():
            raise ValueError(f"Demo file already exists (will not overwrite): {target}")
        source = files("kaggle_gpu_mcp").joinpath(f"demo_{part}.py.txt")
        with source.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return {"prep_code": "demo/prep.py", "train_code": "demo/train.py",
            "directory": str(dest)}
