"""Bounded, read-only CSV profiling and provenance hashing.

Registration never fetches untrusted URLs or sends local data to Kaggle.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile_csv(path: Path, max_rows: int = 1_000_000) -> dict:
    """Stream a CSV; return quality metadata, not potentially private row contents."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not columns or len(columns) > 200 or any(not c or len(c) > 256 for c in columns):
            raise ValueError("CSV requires 1-200 valid column headers")
        if len(set(columns)) != len(columns):
            raise ValueError("CSV has duplicate column names")
        missing = {col: 0 for col in columns}
        numeric = {col: 0 for col in columns}
        nonempty = {col: 0 for col in columns}
        count = 0
        for row in reader:
            count += 1
            if count > max_rows:
                raise ValueError("CSV exceeds bounded profiling row limit")
            if None in row:  # too many fields
                raise ValueError("CSV contains rows wider than its schema")
            for column in columns:
                value = row[column]
                if value is None or not value.strip():
                    missing[column] += 1
                else:
                    nonempty[column] += 1
                    try:
                        float(value)
                    except (ValueError, OverflowError):
                        pass
                    else:
                        numeric[column] += 1
    return {
        "columns": columns,
        "row_count": count,
        "missing_values": missing,
        "numeric_columns": [c for c in columns if nonempty[c] and numeric[c] == nonempty[c]],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
