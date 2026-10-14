"""Untrusted AI requests must not select arbitrary private files or inject CLI arguments."""
import re
from pathlib import Path

USER = re.compile(r"^[a-zA-Z0-9_-]{2,40}$")
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{2,65}$")
REF = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+(?:/[0-9]+)?$")


def validate_username(name: str) -> str:
    if not USER.fullmatch(name):
        raise ValueError("Set a valid KGM_KAGGLE_USERNAME before submitting jobs")
    return name


def validate_slug(slug: str) -> str:
    if not SLUG.fullmatch(slug):
        raise ValueError("Slug must be 3–66 characters: lowercase letters, digits, dashes")
    return slug


def validate_ref(ref: str) -> str:
    if not REF.fullmatch(ref):
        raise ValueError(f"Invalid Kaggle reference: {ref!r}")
    return ref


def allow_file(workspace: Path, candidate: str, suffixes: tuple[str, ...]) -> Path:
    root = workspace.resolve()
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("File does not exist") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("File must exist inside the configured workspace")
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(f"Expected one of {suffixes}")
    if resolved.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("Code file exceeds 25 MiB limit")
    return resolved


def allow_dir(workspace: Path, candidate: str) -> Path:
    root = workspace.resolve()
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Directory does not exist") from exc
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise ValueError("Directory must exist inside configured workspace")
    return resolved


def validate_sources(sources: list[str]) -> list[str]:
    if len(sources) > 25:
        raise ValueError("Too many input sources")
    return list(dict.fromkeys(validate_ref(s) for s in sources))
